# -*- coding: utf-8 -*-
"""
FastAPI + LangChain v1 + PostgreSQL/PGVector

Migração:
- AgentExecutor + create_react_agent + Tool -> create_agent + @tool
- PromptTemplate ReAct -> system_prompt
- input/chat_history/agent_scratchpad -> messages
- response["output"] -> result["messages"][-1].content
"""

from datetime import datetime
import io
import json
import os
import ast
import operator
import psycopg2
import psycopg2.extras

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

data_hora_atual = datetime.now()
data_hora_string = data_hora_atual.strftime("%d/%m/%Y %H:%M:%S")

#Coloque a API KEY da openAi na variavel de ambiente OPENAI_API_KEY
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

os.environ["IS_GAME"] = "False"

DB_USER = os.getenv("PG_USER", "mateus")
DB_PASSWORD = os.getenv("PG_PASSWORD", "mateusmelo95")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DB", "langchain")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

CONNECTION_STRING = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# =============================================================================
# BANCO POSTGRES
# =============================================================================

def get_psycopg_conn():
    return psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def ensure_user_files_table(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_pdf_files (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            title TEXT,
            upload_time TIMESTAMPTZ DEFAULT now(),
            size BIGINT
        );
        """
    )
    conn.commit()
    cur.close()


def save_file_to_db(conn, filename, pdf_bytes, title=None):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_pdf_files
            ( filename, title, upload_time, size)
        VALUES (%s, %s, now(), %s)
        RETURNING id;
        """,
        (filename, title, len(pdf_bytes)),
    )
    file_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return file_id


def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def process_and_store_pdf_bytes(pdf_bytes, user_id, book_title, filename):
    full_text = extract_text_from_pdf_bytes(pdf_bytes)

    if not full_text.strip():
        return {
            "chunks": 0,
            "warning": "PDF sem texto extraído (talvez seja imagem/scan).",
        }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_text(full_text)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "book_title": book_title or filename,
                "chunk_id": i,
                "source_file": filename,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    collection_name = f"book_chunks"

    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection=CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=False,
    )

    return {
        "chunks": len(documents),
        "collection": collection_name,
    }



# =============================================================================
# FERRAMENTAS LANGCHAIN v1
# =============================================================================

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}



@tool
def busca_vetorial(query: str) -> str:
    """Busca trechos semelhantes nos PDFs do usuário usando PGVector."""
    try:
        collection_name = f"book_chunks"

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )

        results = vectorstore.similarity_search(query, k=5)

        if not results:
            return "Nenhum resultado relevante encontrado."

        formatted_results = "\n".join(
            [
                f"Chunk: {doc.page_content}\n"
                f"Metadados: {doc.metadata}"
                for doc in results
            ]
        )

        return (
            f"Resultados da busca vetorial para '{query}':\n"
            f"{formatted_results}"
        )

    except Exception as e:
        return f"Erro na busca vetorial: {str(e)}"


tools = [
    busca_vetorial,
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================


prompt_template = """



## REGRAS

Você é um auditor do Tribunal de Contas da União e deve analisar os riscos da obra de uma rodovia, seu nome é Julia.


Seguem as regras do cruzamento de informações das geometrias fornecidas:

1) Interseção de Rodovia com Unidade de Conservação:
indica quais unidades de conservação ambiental existem num buffer de 40 quilômetros da rodovia. O atributo areaSobreposição indica a área de sobreposição entre a unidade de conservação e o buffer da rodovia em hectares. O atributo areahaalb indica a área total da unidade de conservação, também em hectares. A proporção entre ambos atributos é relevante para avaliação do impacto ambiental. Use o nome da unidade, no atributo nomeuc, para buscar elementos adicionais sobre as características de cada unidade para avaliar como a rodovia e sua obra podem afeta-la. A Lei do SNUC (Lei Federal nº 9.985/2000) que Institui o Sistema Nacional de Unidades de Conservação exige autorização do órgão responsável pela administração da unidade de conservação para qualquer empreendimento de significativo impacto ambiental, estabelecendo restrições específicas dependendo se a UC é de Proteção Integral ou de Uso Sustentável. A Política Nacional do Meio Ambiente (Lei Federal nº 6.938/1981) estabelece o licenciamento ambiental e a obrigatoriedade da reparação de danos ambientais.A Resolução CONAMA nº 01/198 exige a elaboração do EIA/RIMA (Estudo e Relatório de Impacto Ambiental) para a construção de estradas de rodagem com duas ou mais faixas de rolamento. A Resolução CONAMA nº 237/1997: Regulamenta os procedimentos gerais para o Licenciamento Ambiental, dividindo as licenças em Prévia (LP), de Instalação (LI) e de Operação (LO).  

2)Interseção de Rodovia com Município:
relaciona os municípios atravessados pela rodovia. Obtenha a população desses municípios para indicar quantas pessoas devem ser afetadas direta ou indiretamente pela obra.

3)Interseção de Rodovia com Áreas Urbanas:
indica quais áreas urbanas vão ser cruzadas pela rodovia. O atributo area_km2 indica a área ocupada por cada uma em quilômetros quadrados, o que dá uma noção do impacto da rodovia no local. O projeto deve avaliar desapropriações e deslocamento de comunidades, além de integração a vias locais e passagens de pedestres nessas áreas.  

4)Interseção de Rodovia com Aldeias:
lista as aldeias indígenas a uma distância de até 40 quilômetros da rodovia. Com base na Convenção 169 da Organização Internacional do Trabalho (OIT), o Estado tem a obrigação de consultar as comunidades afetadas antes de aprovar medidas legislativas ou administrativas que as impactem. As comunidades devem ser consultadas de acordo com seus próprios protocolos de consulta, que são documentos elaborados pelos indígenas para ditar como desejam ser ouvidos e respeitados durante o processo de decisão sobre a obra. Os resultados das etapas técnicas devem ser apresentados aos indígenas em reuniões específicas e, quando necessário, em materiais traduzidos para as línguas nativas.

5)Interseção de Rodovia com Floresta Pública:
indica quais florestas públicas existem num buffer de 40 quilômetros da rodovia. O atributo areaSobreposição indica a área de sobreposição entre a unidade de conservação e o buffer da rodovia em hectares. O atributo area_ha indica a área total da unidade de conservação, também em hectares. A proporção entre ambos atributos é relevante para avaliação do impacto ambiental.A construção ou pavimentação de rodovias em florestas públicas e Unidades de Conservação (UCs) exige obrigatoriamente Licenciamento Ambiental Federal, aprovação dos órgãos gestores (como o ICMBio).

6)Interseção de Rodovia com Sitios Arqueológicos:
relaciona os sítios arqueológicos a uma distância de até 40 quilômetros da rodovia. A construção de rodovias com sítios arqueológicos na faixa de domínio exige licenciamento ambiental e anuência do IPHAN. Qualquer dano ao patrimônio é crime federal. É obrigatório realizar prospecção, isolamento, monitoramento e, se necessário, resgate dos vestígios por arqueólogos antes da obra. O IPHAN avalia o impacto e emite a Portaria de Autorização para pesquisa arqueológica.  A área do sítio não pode ser terraplenada ou utilizada para bota-fora, empréstimo de terra ou trânsito de maquinário sem autorização prévia.  Durante a limpeza do terreno e movimentação de terra, um arqueólogo deve acompanhar as frentes de serviço para identificar novos vestígios ocultos.  Caso o sítio não possa ser evitado pelo traçado da rodovia, os arqueólogos realizam a escavação controlada para a retirada e catalogação dos artefatos antes da pavimentação.  Se vestígios forem encontrados durante a obra, os trabalhos devem ser paralisados imediatamente no trecho afetado e o IPHAN acionado.Para verificar se o trecho onde você está atuando já possui algum sítio mapeado ou para formalizar o licenciamento, você pode consultar o Protocolo Digital do IPHAN ou avaliar os estudos previstos nas diretrizes da IN IPHAN nº 6/2025.

7)Interseção de Rodovia com Terras Indígenas:
indica as terras indígenas que serão cruzadas pelo rodovia. O atributo superfície indica a área total da reserva em hectares e o campo areaSobreposiçãoha indica a área de sobreposição entre a terra indígena e um buffer de 40 quilômetros ao longo da rodovia, também em hectares. A proporção entre essas duas áreas é uma informação relevante do impacto da rodovia sobre a reserva indígena.

Analise os potenciais riscos da obra e indique verificações que o auditor deve fazer com relação às ciscunstâncias da obra e gere um relatório em HTML sem CSS externo. Responda apenas com o o relatório em HTML. Utilize: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <table>."    


# INFORMAÇÕES ADICIONAIS 

A data e hora atual é {{{data_atual}}}.


## Ferramentas
    
    #calculadora
        - Use esssa ferramenta para fazer cálculos (ex.: 2+2, 5*3).
    #elevar
        - Use essa ferramenta para elevar ou fazer uma exponeciação (ex.: 2**2, 2^2).


    #busca_vetorial
    - Use essa ferramenta para buscar no banco de dados sobre um tema, passe para a ferramenta o título do livro e o conteúdo desejado.

    

Ferramentas disponíveis:
{tools}

Nomes das ferramentas:
{tool_names}

Histórico da conversa:
{chat_history}

Pergunta:
{input}

Raciocínio até aqui:
{agent_scratchpad}
"""

system_prompt = prompt_template


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/webhook")
async def handle_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    book_title: str = Form(None),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Arquivo deve ser PDF",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Max {MAX_UPLOAD_MB} MB",
        )

    conn = get_psycopg_conn()

    try:
        ensure_user_files_table(conn)

        file_id = save_file_to_db(
            conn,
            user_id,
            file.filename,
            pdf_bytes,
            book_title or file.filename,
        )
    finally:
        conn.close()

    try:
        result = process_and_store_pdf_bytes(
            pdf_bytes,
            user_id,
            book_title or file.filename,
            file.filename,
        )
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "file_id": file_id,
        }

    return {
        "status": "ok",
        "file_id": file_id,
        "vector_result": result,
    }


@app.post("/webhook")
async def relatorio_webhook(request: Request):


    data = await request.json()

    # No código original data.get("chatInput") era chamado, mas o resultado
    # era descartado. Aqui a mensagem realmente é usada.
    user_message = data.get("chatInput", "")

    if not user_message:
        user_message = data.get(
            "message",
            data.get("input", ""),
        )

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Campo 'chatInput' não informado.",
        )

    data_hora_atual = datetime.now()
    data_hora_string = data_hora_atual.strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    # O prompt do agente v1 é um system prompt normal.
    # Fazemos apenas a substituição das variáveis que existiam no código antigo.
    system_prompt_runtime = (
        system_prompt
        .replace("{{{data_atual}}}", data_hora_string)
    )


    try:

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt_runtime,
        )

        entrada_formatada = (
            f"{user_message}\n"
            f"(Data atual: {data_hora_string})"
        )

        # LangChain v1 usa a sequência de mensagens como entrada.
        messages = [
            HumanMessage(content=entrada_formatada)
        ]

        result = agent.invoke(
            {"messages": messages},
            config={
                # Limita o número de ciclos do grafo/agente.
                "recursion_limit": 25
            },
        )

        result_messages = result.get("messages", [])

        if not result_messages:
            raise RuntimeError(
                "O agente não retornou mensagens."
            )

        final_message = result_messages[-1]
        agent_output = final_message.content

        if not isinstance(agent_output, str):
            agent_output = json.dumps(
                agent_output,
                ensure_ascii=False,
                default=str,
            )

        print(f"Agente: {agent_output}\n")


        return {
            "status": "received",
            "agent_output": agent_output,
        }

    except Exception as e:
        print(f"Erro no agente: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar agente: {str(e)}",
        )



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5679,
        workers=1,
    )
