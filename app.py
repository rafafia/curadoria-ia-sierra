
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Catálogo Inteligente Sierra – Curadoria IA")

st.markdown("Descreva o ambiente ou estilo desejado e selecione múltiplas categorias. A IA irá sugerir os produtos mais compatíveis com base nas descrições visuais.")

# Carregar o catálogo atualizado com links do GitHub
catalogo = pd.read_excel("catalogo_sierra_30produtos_github.xlsx")

# Upload de imagem de referência (opcional)
imagem_cliente = st.file_uploader("📸 Envie uma imagem de referência do ambiente (opcional)", type=["jpg", "jpeg", "png"])

# Entrada do usuário
descricao_cliente = st.text_area("📝 Descreva o ambiente, estilo ou o que o cliente busca:")

# Seleção de múltiplas categorias
categorias = st.multiselect("📂 Selecione as categorias desejadas:", catalogo['Categoria'].unique())
gerar = st.button("Gerar orçamento")

if gerar and descricao_cliente and categorias:
    st.subheader("🛋️ Produtos Sugeridos (Top 3 mais compatíveis)")
    
    produtos_filtrados = catalogo[catalogo['Categoria'].isin(categorias)].reset_index(drop=True)
    corpus = [descricao_cliente] + produtos_filtrados['Descrição Visual'].tolist()
    tfidf = TfidfVectorizer().fit_transform(corpus)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    top_indices = similarity.argsort()[-3:][::-1]

    for idx in top_indices:
        produto = produtos_filtrados.iloc[idx]
        st.image(produto['Imagem'], width=300)
        st.markdown(f"**Nome:** {produto['Nome']}")
        st.markdown(f"**Categoria:** {produto['Categoria']}")
        st.markdown(f"**Descrição Técnica:** {produto['Descrição Técnica']}")
        st.markdown(f"**Descrição Visual:** {produto['Descrição Visual']}")
        st.markdown(f"**Medidas:** {produto['Medidas (cm)']}")
        st.markdown(f"**Módulos Disponíveis:** {produto['Módulos Disponíveis']}")
        st.markdown("---")

    st.success("✅ Orçamento com múltiplos produtos sugeridos com base na sua descrição.")

    if imagem_cliente:
        st.markdown("🔍 **Imagem enviada como referência:**")
        st.image(imagem_cliente, width=250)

st.markdown("---")
st.caption("Desenvolvido por Rafael Ferrer – IA para curadoria de produtos de alto padrão.")
