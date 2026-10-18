import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==========================================
# إعداد الصفحة
# ==========================================
st.set_page_config(
    page_title="مستكشف المسار الوظيفي",
    page_icon="🔎",
    layout="centered"
)


# ==========================================
# CSS للواجهة العربية
# ==========================================
st.markdown(
    """
    <style>

    .stApp {
        direction: rtl;
    }

    h1 {
        text-align: center;
    }

    p {
        text-align: right;
    }

    textarea {
        direction: rtl !important;
        text-align: right !important;
        font-size: 16px !important;
    }

    .stTextArea label {
        direction: rtl !important;
        text-align: right !important;
    }

    .stButton button {
        width: 100%;
        font-size: 17px;
    }

    .result-box {
        direction: rtl;
        text-align: right;
        padding: 20px;
        border: 1px solid #444;
        border-radius: 12px;
        margin-top: 20px;
        line-height: 2.2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# تحميل وتنظيف البيانات
# ==========================================
@st.cache_data
def load_data():

    df = pd.read_csv("Jadarat_data.csv")

    # حذف الصفوف المكررة
    df = df.drop_duplicates()

    # معالجة القيم الناقصة وتنظيف الفراغات
    df["job_title"] = (
        df["job_title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["job_desc"] = (
        df["job_desc"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["job_tasks"] = (
        df["job_tasks"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # نحتفظ فقط بالأعمدة المهمة
    jobs = df[
        [
            "job_title",
            "job_desc",
            "job_tasks"
        ]
    ].copy()

    # حذف الوظائف التي لا تحتوي على مسمى
    jobs = jobs[
        jobs["job_title"] != ""
    ].reset_index(drop=True)

    return jobs


jobs = load_data()


# ==========================================
# تحميل نموذج Embeddings
# ==========================================
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return model


model = load_model()


# ==========================================
# حساب Embeddings للوظائف مرة واحدة
# ==========================================
@st.cache_resource
def create_job_embeddings():

    title_embeddings = model.encode(
        jobs["job_title"].tolist(),
        show_progress_bar=False
    )

    tasks_embeddings = model.encode(
        jobs["job_tasks"].tolist(),
        show_progress_bar=False
    )

    desc_embeddings = model.encode(
        jobs["job_desc"].tolist(),
        show_progress_bar=False
    )

    return (
        title_embeddings,
        tasks_embeddings,
        desc_embeddings
    )


(
    title_embeddings,
    tasks_embeddings,
    desc_embeddings
) = create_job_embeddings()


# ==========================================
# دالة اقتراح الوظائف
# ==========================================
def recommend_jobs(user_text):

    if not user_text or not user_text.strip():
        return []

    # تحويل كلام المستخدم إلى Embedding
    user_embedding = model.encode(
        [user_text],
        show_progress_bar=False
    )

    # التشابه مع المسمى الوظيفي
    title_similarity = cosine_similarity(
        user_embedding,
        title_embeddings
    )[0]

    # التشابه مع المهام
    tasks_similarity = cosine_similarity(
        user_embedding,
        tasks_embeddings
    )[0]

    # التشابه مع وصف الوظيفة
    desc_similarity = cosine_similarity(
        user_embedding,
        desc_embeddings
    )[0]

    # ======================================
    # حساب الدرجة النهائية
    #
    # المسمى = 30%
    # المهام = 50%
    # الوصف = 20%
    # ======================================
    final_similarity = (
        title_similarity * 0.30
        + tasks_similarity * 0.50
        + desc_similarity * 0.20
    )

    # إنشاء نسخة مؤقتة للنتائج
    results_df = jobs.copy()

    results_df["final_similarity"] = final_similarity

    # ترتيب الوظائف من الأعلى إلى الأقل
    results_df = results_df.sort_values(
        by="final_similarity",
        ascending=False
    )

    # إزالة تكرار نفس المسمى الوظيفي
    results_df = results_df.drop_duplicates(
        subset=["job_title"]
    )

    # أخذ أفضل 5 نتائج
    top_5 = results_df.head(5)

    results = []

    for _, row in top_5.iterrows():

        job_title = row["job_title"]

        score = round(
            float(row["final_similarity"]) * 100,
            2
        )

        results.append(
            {
                "title": job_title,
                "score": score
            }
        )

    return results


# ==========================================
# واجهة المستخدم
# ==========================================
st.title("مستكشف المسار الوظيفي")

st.markdown(
    """
    <div style="
        text-align: center;
        direction: rtl;
        margin-bottom: 25px;
    ">
        اكتب الأشياء التي تحب القيام بها،
        وسنقترح لك أقرب المسميات الوظيفية.
    </div>
    """,
    unsafe_allow_html=True
)


user_text = st.text_area(
    "اكتب الأشياء اللي تحب تسويها",
    placeholder=(
        "مثال: أحب تحليل البيانات والعمل "
        "مع قواعد البيانات وحل المشاكل"
    ),
    height=180
)


# ==========================================
# زر الاقتراح
# ==========================================
if st.button(
    "اقترح لي وظائف",
    type="primary",
    use_container_width=True
):

    if not user_text.strip():

        st.warning(
            "اكتب وصفًا للأشياء التي تحب القيام بها أولًا."
        )

    else:

        with st.spinner("جاري البحث عن الوظائف المناسبة..."):

            results = recommend_jobs(user_text)

        # عرض النتائج
        st.subheader("الوظائف المقترحة")

        with st.container(border=True):

            for number, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"""
                    <div style="
                        direction: rtl;
                        text-align: right;
                        font-size: 18px;
                        margin-bottom: 10px;
                    ">
                        <strong>{number}. {result["title"]}</strong>
                        — {result["score"]}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )