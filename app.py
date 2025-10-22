# ==============================================================================
# GC-MS Data Comparator & Flavor Explorer
# Complete Version with Integrated Flavor Database (No Word Cloud)
# Created by Aniwat Kaewkrod
# Bug Fixes Applied by Manus
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import os
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# ส่วนที่ 1: ฟังก์ชันจัดการฐานข้อมูลกลิ่น
# ==============================================================================

@st.cache_data
def get_flavor_database():
    """
    อ่านและโหลดฐานข้อมูลกลิ่นจากไฟล์
    """
    try:
        all_dfs = []
        file_stats = []
        
        flavor_files = [
            'flavordb_descriptive.csv',
            'flavornet_descriptive.csv',
            'flavor_descriptive_master.csv'
        ]
        
        for file_path in flavor_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    
                    df.columns = [str(col).strip().lower() for col in df.columns]
                    
                    possible_name_cols = ['compound_name', 'compound']
                    possible_flavor_cols = ['flavor_description', 'odor_description', 'flavor', 'descriptors']
                    possible_source_cols = ['source']

                    name_col = next((col for col in possible_name_cols if col in df.columns), None)
                    flavor_col = next((col for col in possible_flavor_cols if col in df.columns), None)
                    source_col = next((col for col in possible_source_cols if col in df.columns), None)

                    temp_df = pd.DataFrame()

                    if name_col and flavor_col:
                        if source_col:
                            temp_df = df[[name_col, flavor_col, source_col]]
                            temp_df.columns = ['Compound', 'Flavor_Descriptor', 'Source']
                        else:
                            temp_df = df[[name_col, flavor_col]]
                            temp_df.columns = ['Compound', 'Flavor_Descriptor']
                            temp_df['Source'] = os.path.splitext(os.path.basename(file_path))[0]
                    
                    if temp_df.empty:
                        st.warning(f"⚠️ ข้ามไฟล์ {file_path} เนื่องจากไม่พบคอลัมน์ที่ต้องการ (ตรวจสอบชื่อคอลัมน์ในไฟล์ CSV)")
                        continue

                    temp_df.dropna(subset=['Compound', 'Flavor_Descriptor'], inplace=True)
                    temp_df['Compound'] = temp_df['Compound'].astype(str).str.strip()
                    temp_df['Flavor_Descriptor'] = temp_df['Flavor_Descriptor'].astype(str).str.strip()
                    
                    temp_df = temp_df[temp_df['Compound'].str.len() > 2]
                    temp_df = temp_df[temp_df['Flavor_Descriptor'].str.len() > 2]
                    
                    file_stats.append({'file': file_path, 'count': len(temp_df)})
                    all_dfs.append(temp_df)
                    
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถอ่านไฟล์ {file_path}: {e}")
                    continue
        
        if not all_dfs:
            st.warning("⚠️ ไม่พบไฟล์ฐานข้อมูลกลิ่น - ฟีเจอร์ Flavor Profile และ Flavor Explorer จะไม่สามารถใช้งานได้")
            return pd.DataFrame(columns=['Compound', 'Flavor_Descriptor', 'Source'])
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['Compound'] = combined_df['Compound'].str.title()
        combined_df = combined_df.drop_duplicates(subset=['Compound'], keep='first')
        
        st.success(f"✅ โหลดฐานข้อมูลกลิ่นสำเร็จ: {len(combined_df):,} สารประกอบ")
        
        with st.expander("📊 รายละเอียดฐานข้อมูล"):
            for stat in file_stats:
                st.write(f"- **{stat['file']}**: {stat['count']:,} สาร")
            st.write(f"- **รวมทั้งหมด (หลังลบซ้ำ)**: {len(combined_df):,} สาร")
        
        return combined_df
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ฐานข้อมูลกลิ่น: {e}")
        return pd.DataFrame(columns=['Compound', 'Flavor_Descriptor', 'Source'])

# ==============================================================================
# ส่วนที่ 2: ฟังก์ชันประมวลผลข้อมูล GC-MS
# ==============================================================================

def clean_compound_name(name):
    if not isinstance(name, str):
        return 'Unknown'
    cleaned = name.strip().strip('"').strip("'")
    cleaned = re.sub(r'^\d+[\s\-\.]+', '', cleaned)
    cleaned = re.sub(r'^\([+\-RSEZ]+\)[\s\-\.]*', '', cleaned)
    cleaned = re.sub(r'^\(\d*[RSEZ][\d,RSEZ]*\)[\s\-\.]*', '', cleaned)
    cleaned = re.sub(r'\s*\(\d+\)\s*$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned.isupper() and len(cleaned) > 3:
        cleaned = cleaned.title()
    cleaned = cleaned.strip()
    if not cleaned or len(cleaned) < 3:
        return 'Unknown'
    return cleaned

def parse_report_file(uploaded_file):
    try:
        sample_name = uploaded_file.name
        file_content = uploaded_file.getvalue().decode("utf-8", errors='ignore')
        lines = [line.strip() for line in file_content.splitlines()]

        peak_list_header_idx = -1
        library_search_header_idx = -1

        for i, line in enumerate(lines):
            if line.startswith('"Peak","R.T."'):
                peak_list_header_idx = i
            elif line.startswith('"PK","RT"'):
                library_search_header_idx = i

        if peak_list_header_idx == -1:
            return None

        header_peaks = [col.strip('"') for col in lines[peak_list_header_idx].split(',')]
        data_peaks = []
        end_idx = library_search_header_idx if library_search_header_idx != -1 else len(lines)
        for i in range(peak_list_header_idx + 1, end_idx):
            if lines[i]:
                parts = lines[i].split(',')
                if parts and parts[0].strip().replace('"', '').isdigit():
                    data_peaks.append(parts)
        
        if not data_peaks:
            return None

        df_peaks = pd.DataFrame(data_peaks)
        num_cols = min(len(df_peaks.columns), len(header_peaks))
        df_peaks = df_peaks.iloc[:, :num_cols]
        df_peaks.columns = header_peaks[:num_cols]

        df_library = None
        if library_search_header_idx != -1:
            header_lib = [col.strip('"') for col in lines[library_search_header_idx].split(',')]
            data_lib = []
            for i in range(library_search_header_idx + 1, len(lines)):
                if lines[i]:
                    parts = lines[i].split(',')
                    if parts and parts[0].strip().replace('"', '').isdigit():
                        data_lib.append(parts)
            if data_lib:
                df_library = pd.DataFrame(data_lib)
                num_cols_lib = min(len(df_library.columns), len(header_lib))
                df_library = df_library.iloc[:, :num_cols_lib]
                df_library.columns = header_lib[:num_cols_lib]

        for col in ["Peak", "R.T.", "Height", "Area"]:
            if col in df_peaks.columns:
                df_peaks[col] = pd.to_numeric(df_peaks[col].astype(str).str.replace('"', ''), errors='coerce')
        df_peaks.dropna(subset=['Peak', 'R.T.'], inplace=True)

        if df_library is not None and 'PK' in df_library.columns:
            df_library['PK'] = pd.to_numeric(df_library['PK'].astype(str).str.replace('"', ''), errors='coerce')
            df_merged = pd.merge(df_peaks, df_library.rename(columns={'PK': 'Peak'}), on='Peak', how='left')
        else:
            df_merged = df_peaks.copy()

        required_cols = ['Library/ID', 'CAS#', 'SI', 'Qual']
        for col in required_cols:
            if col not in df_merged.columns:
                df_merged[col] = 'Unknown'
        df_merged['Library/ID'] = df_merged['Library/ID'].fillna('Unknown')
        df_merged['Sample'] = sample_name
        return df_merged
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ {uploaded_file.name}: {str(e)}")
        return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

# ==============================================================================
# ส่วนที่ 3: การตั้งค่าหน้าและ UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="GC-MS Data Comparator", page_icon="🧪")
st.title("🧪 Multi-Sample GC-MS Data Comparator & Flavor Explorer")
st.markdown('<p style="color:green; font-weight:bold; font-size:16px;">Created by Aniwat Kaewkrod</p>', unsafe_allow_html=True)
st.write("อัปโหลดไฟล์ Report จาก GC-MS (สูงสุด 20 ไฟล์) เพื่อทำการวิเคราะห์และเปรียบเทียบข้อมูล")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

# ==============================================================================
# ส่วนที่ 4: Sidebar - File Upload & Control
# ==============================================================================

st.sidebar.header("📁 File Upload & Control")

# === FIX: เพิ่มคำอธิบายและเน้นข้อความเกี่ยวกับประเภทไฟล์ ===
st.sidebar.markdown("**1. อัปโหลดไฟล์ Report (.CSV หรือ .TXT)**")
st.sidebar.info(
    """
    โปรแกรมรองรับไฟล์ Report ที่ Export จากเครื่อง GC-MS 
    ซึ่งมีโครงสร้างเป็นไฟล์ข้อความ (Text File) ที่ประกอบด้วย:
    - **Peak List** (ต้องมี)
    - **Library Search** (ถ้ามี)
    """
)
uploaded_files = st.sidebar.file_uploader(
    "เลือกไฟล์ Report ของคุณ",
    type=['csv', 'txt'],
    accept_multiple_files=True,
    key="file_uploader",
    label_visibility="collapsed" # ซ่อน Label เพราะมีหัวข้อแล้ว
)
# === END FIX ===

st.sidebar.markdown("---")
st.sidebar.markdown("**2. เริ่มการวิเคราะห์**")
col1, col2 = st.sidebar.columns(2)

analyze_button = col1.button("🚀 Analyze", use_container_width=True, type="primary")
clear_button = col2.button("🗑️ Clear Data", use_container_width=True)

if clear_button:
    st.session_state.analysis_complete = False
    st.session_state.uploaded_files_list = []
    st.rerun()

if analyze_button and uploaded_files:
    if len(uploaded_files) > 20:
        st.error("❌ สามารถเปรียบเทียบได้สูงสุดครั้งละ 20 ไฟล์เท่านั้น")
    else:
        st.session_state.uploaded_files_list = uploaded_files
        st.session_state.analysis_complete = True
        st.rerun()
elif analyze_button and not uploaded_files:
    st.warning("⚠️ กรุณาอัปโหลดไฟล์ก่อนกด Analyze")

# ==============================================================================
# ส่วนที่ 5: Main Display Area
# ==============================================================================

if not st.session_state.analysis_complete:
    st.info("📌 กรุณาอัปโหลดไฟล์ในแถบด้านข้าง แล้วกดปุ่ม 'Analyze' เพื่อเริ่มต้น")
    if uploaded_files:
        st.write("**ไฟล์ที่พร้อมสำหรับการวิเคราะห์:**")
        for idx, f in enumerate(uploaded_files, 1):
            st.write(f"{idx}. ✓ {f.name}")

if st.session_state.analysis_complete and st.session_state.uploaded_files_list:
    with st.spinner("⏳ กำลังประมวลผลข้อมูล... กรุณารอสักครู่"):
        all_data_list = [parse_report_file(f) for f in st.session_state.uploaded_files_list]
        valid_data_list = [df for df in all_data_list if df is not None]

    if not valid_data_list:
        st.error("❌ ไม่สามารถประมวลผลไฟล์ใดๆ ได้ กรุณาตรวจสอบรูปแบบไฟล์")
        st.session_state.analysis_complete = False
        st.stop()

    combined_df = pd.concat(valid_data_list, ignore_index=True)
    combined_df['Compound'] = combined_df['Library/ID'].apply(clean_compound_name)
    base_analysis_df = combined_df[combined_df['Compound'] != 'Unknown'].copy()

    num_samples = len(valid_data_list)
    num_compounds = len(base_analysis_df['Compound'].unique())
    num_peaks = len(base_analysis_df)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("📊 จำนวนไฟล์", f"{num_samples} ไฟล์")
    col_stat2.metric("🧪 จำนวนสาร", f"{num_compounds} สาร")
    col_stat3.metric("📈 จำนวน Peaks", f"{num_peaks} peaks")
    st.success("✅ ประมวลผลสำเร็จ!")

    tabs = st.tabs(["🔥 Comparative Heatmap", "🧬 PCA Clustering", "👃 Flavor Profile", "🔍 Flavor Explorer", "📊 Overlaid Chromatograms", "📄 Combined Data"])

    with tabs[0]:
        st.header("🔥 เปรียบเทียบปริมาณสารระหว่างตัวอย่าง")
        st.subheader("⚙️ Display Options")
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        value_option = opt_col1.selectbox("เลือกค่าที่จะแสดง:", ("Area", "Height"), key="heatmap_value")
        colorscale_option = opt_col2.selectbox("เลือกชุดโทนสี:", ("Plasma", "Viridis", "Cividis", "Blues", "Reds", "Greens", "YlGnBu", "YlOrRd", "Inferno", "Magma", "Turbo"), index=0)
        filter_contaminants = opt_col3.checkbox("กรองสาร Siloxanes", value=True, key="filter_check")
        use_log_scale = opt_col4.checkbox("ใช้สเกลสีแบบ Log", value=False, key="log_scale_check")
        st.markdown("---")
        st.subheader("🔎 Filter & Sort")
        filter_col1, filter_col2 = st.columns(2)
        search_term = filter_col1.text_input("ค้นหาสารประกอบ:", "", placeholder="พิมพ์ชื่อสารที่ต้องการค้นหา...").strip().lower()
        sort_option = filter_col2.selectbox("เรียงลำดับสารประกอบ:", ("Alphabetical (A-Z)", "Total Abundance (Highest First)", "Variance (Highest First)"), key="compound_sort")

        CONTAMINANT_KEYWORDS = ['siloxane', 'cyclotrisiloxane', 'cyclopentasiloxane', 'cyclotetrasiloxane', 'cyclohexasiloxane', 'cyclododecasiloxane']
        contaminant_pattern = '|'.join(CONTAMINANT_KEYWORDS)
        heatmap_df = base_analysis_df.copy()
        if filter_contaminants:
            heatmap_df = heatmap_df[~heatmap_df['Compound'].str.contains(contaminant_pattern, case=False, na=False)]
        if search_term:
            heatmap_df = heatmap_df[heatmap_df['Compound'].str.lower().str.contains(search_term, na=False)]

        if not heatmap_df.empty:
            try:
                heatmap_pivot = heatmap_df.pivot_table(index='Compound', columns='Sample', values=value_option, aggfunc='sum').fillna(0)
                heatmap_pivot = heatmap_pivot[heatmap_pivot.sum(axis=1) > 0]
                if not heatmap_pivot.empty:
                    if sort_option == "Total Abundance (Highest First)":
                        heatmap_pivot = heatmap_pivot.loc[heatmap_pivot.sum(axis=1).sort_values(ascending=False).index]
                    elif sort_option == "Variance (Highest First)":
                        heatmap_pivot = heatmap_pivot.loc[heatmap_pivot.var(axis=1).sort_values(ascending=False).index]
                    else:
                        heatmap_pivot = heatmap_pivot.sort_index(ascending=True)

                    plot_data = np.log1p(heatmap_pivot) if use_log_scale else heatmap_pivot
                    color_label = f"Log({value_option})" if use_log_scale else value_option
                    graph_height = max(600, min(len(plot_data.index) * 25, 3000))

                    fig_heatmap = px.imshow(plot_data, labels=dict(x="Sample (ตัวอย่าง)", y="Compound (สาร)", color=color_label), aspect="auto", color_continuous_scale=colorscale_option, height=graph_height)
                    fig_heatmap.update_layout(xaxis_side="top", xaxis=dict(tickangle=-45), yaxis=dict(tickmode='linear', automargin=True), font=dict(size=10), margin=dict(l=200, r=50, t=100, b=50))
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    st.caption("📖 **วิธีอ่าน Heatmap:**\n- **สีเข้ม (น้ำเงิน/ม่วง):** ปริมาณสาร **น้อย**\n- **สีอ่อน (เหลือง/แดง):** ปริมาณสาร **มาก**\n- **แกน X:** ชื่อไฟล์ตัวอย่าง\n- **แกน Y:** ชื่อสารประกอบ")
                    st.markdown("---")
                    st.subheader("💾 Export Options")
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        html_bytes = fig_heatmap.to_html(include_plotlyjs='cdn').encode()
                        st.download_button("📥 Download Heatmap (HTML)", html_bytes, "heatmap.html", "text/html", use_container_width=True, help="ดาวน์โหลดเป็นไฟล์ HTML แบบ Interactive (เปิดด้วย Browser)")
                    with exp_col2:
                        st.download_button("📥 Download Data (XLSX)", to_excel(heatmap_pivot.reset_index()), "heatmap_data.xlsx", use_container_width=True)
                else:
                    st.warning("⚠️ ไม่พบข้อมูลที่เพียงพอสำหรับสร้าง Heatmap หลังจากการกรอง")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการสร้าง Heatmap: {e}")
        else:
            st.warning("⚠️ ไม่พบสารที่ตรงกับเงื่อนไขการค้นหา")

    with tabs[1]:
        st.header("🧬 การจัดกลุ่มตัวอย่าง (PCA)")
        st.info("💡 แสดงความคล้ายคลึงกันของโปรไฟล์สารเคมี - ตัวอย่างที่อยู่ใกล้กัน = โปรไฟล์คล้ายกัน")
        if len(base_analysis_df['Sample'].unique()) < 2:
            st.warning("⚠️ ต้องมีตัวอย่างอย่างน้อย 2 ไฟล์เพื่อทำ PCA")
        else:
            try:
                pca_pivot = base_analysis_df.pivot_table(index='Sample', columns='Compound', values='Area', aggfunc='sum').fillna(0)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_pivot)
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_data)
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=pca_pivot.index)
                explained_var = pca.explained_variance_ratio_
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text=pca_df.index, labels={'PC1': f'PC1 ({explained_var[0]*100:.2f}%)', 'PC2': f'PC2 ({explained_var[1]*100:.2f}%)'}, title="2D PCA Clustering of Samples")
                fig_pca.update_traces(textposition='top center', marker=dict(size=15, line=dict(width=2, color='white')))
                fig_pca.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'), yaxis2=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'), height=600)
                st.plotly_chart(fig_pca, use_container_width=True)
                total_var = (explained_var[0] + explained_var[1]) * 100
                st.caption(f"📖 **วิธีอ่าน PCA Plot:**\n- แต่ละจุด = 1 ไฟล์ตัวอย่าง\n- จุดที่ใกล้กัน = โปรไฟล์สารเคมีคล้ายกัน\n- PC1 และ PC2 อธิบายความแปรปรวนรวม **{total_var:.2f}%**")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ PCA: {e}")

    with tabs[2]:
        st.header("👃 สรุปโปรไฟล์กลิ่น")
        st.info("💡 แสดงคำอธิบายกลิ่นของสารประกอบที่ตรวจพบ โดยอ้างอิงจากฐานข้อมูล")
        df_flavor_db = get_flavor_database()
        if not df_flavor_db.empty:
            found_compounds = base_analysis_df[['Compound']].drop_duplicates()
            flavor_profile = pd.merge(found_compounds, df_flavor_db, on='Compound', how='left')
            compounds_with_flavor = flavor_profile.dropna(subset=['Flavor_Descriptor'])
            compounds_without_flavor = flavor_profile[flavor_profile['Flavor_Descriptor'].isna()]
            col_fp1, col_fp2, col_fp3 = st.columns(3)
            col_fp1.metric("🧪 สารทั้งหมด", len(found_compounds))
            col_fp2.metric("✅ มีข้อมูลกลิ่น", len(compounds_with_flavor))
            col_fp3.metric("❌ ไม่มีข้อมูล", len(compounds_without_flavor))
            if not compounds_with_flavor.empty:
                st.markdown("---")
                st.subheader("📋 สารประกอบที่มีข้อมูลกลิ่น")
                st.dataframe(compounds_with_flavor[['Compound', 'Flavor_Descriptor', 'Source']], use_container_width=True, hide_index=True, column_config={"Compound": "ชื่อสาร", "Flavor_Descriptor": "คำอธิบายกลิ่น", "Source": "แหล่งข้อมูล"})
                st.markdown("---")
                st.subheader("📊 สถิติตามแหล่งข้อมูล")
                source_counts = compounds_with_flavor['Source'].value_counts().reset_index()
                source_counts.columns = ['แหล่งข้อมูล', 'จำนวนสาร']
                st.dataframe(source_counts, use_container_width=True, hide_index=True)
                st.markdown("---")
                st.download_button("📥 Download Flavor Profile (XLSX)", to_excel(compounds_with_flavor), "flavor_profile.xlsx", help="ดาวน์โหลดข้อมูลกลิ่นทั้งหมดในรูปแบบ Excel")
            else:
                st.warning("⚠️ ไม่พบสารประกอบใดๆ ที่มีข้อมูลกลิ่นในฐานข้อมูล")
        else:
            st.error("❌ ไม่สามารถโหลดฐานข้อมูลกลิ่นได้ - ฟีเจอร์นี้ไม่สามารถใช้งานได้")

    with tabs[3]:
        st.header("🔍 เครื่องมือค้นหากลิ่นย้อนกลับ")
        st.info("💡 ค้นหาจาก 'กลิ่น' เพื่อดูว่ามีสารใดบ้างที่ให้กลิ่นนั้น และพบในตัวอย่างของคุณหรือไม่")
        df_flavor_db = get_flavor_database()
        if not df_flavor_db.empty:
            search_flavor = st.text_input("🔍 พิมพ์กลิ่นที่ต้องการค้นหา:", "", placeholder="เช่น fruity, nutty, caramel, floral, sweet...").strip().lower()
            if search_flavor:
                try:
                    found_compounds = df_flavor_db[df_flavor_db['Flavor_Descriptor'].str.contains(search_flavor, case=False, na=False)].copy()
                    if not found_compounds.empty:
                        st.subheader(f"📊 พบ {len(found_compounds):,} สารประกอบที่มีกลิ่น '{search_flavor}'")
                        samples_compounds = base_analysis_df['Compound'].unique()
                        found_compounds['Found_In_Samples'] = found_compounds['Compound'].isin(samples_compounds)
                        num_found_in_samples = found_compounds['Found_In_Samples'].sum()
                        if num_found_in_samples > 0:
                            st.success(f"✅ พบ {num_found_in_samples} สารในตัวอย่างของคุณ!")
                        else:
                            st.info("💡 ไม่พบสารที่ให้กลิ่นนี้ในตัวอย่างของคุณ")
                        st.dataframe(found_compounds[['Compound', 'Flavor_Descriptor', 'Source', 'Found_In_Samples']], column_config={"Compound": "ชื่อสาร", "Flavor_Descriptor": "คำอธิบายกลิ่น", "Source": "แหล่งข้อมูล", "Found_In_Samples": st.column_config.CheckboxColumn("✓ พบในตัวอย่าง?", help="สารนี้ถูกตรวจพบในตัวอย่างที่คุณอัปโหลดหรือไม่")}, use_container_width=True, hide_index=True)
                        compounds_to_plot = found_compounds[found_compounds['Found_In_Samples']]['Compound']
                        if not compounds_to_plot.empty:
                            st.markdown("---")
                            st.subheader(f"📈 เปรียบเทียบปริมาณสารที่ให้กลิ่น '{search_flavor}'")
                            plot_data = base_analysis_df[base_analysis_df['Compound'].isin(compounds_to_plot)]
                            if not plot_data.empty:
                                fig_bar = px.bar(plot_data, x="Sample", y="Area", color="Compound", title=f"ปริมาณสารที่ให้กลิ่น '{search_flavor}' ในแต่ละตัวอย่าง", labels={"Sample": "ตัวอย่าง", "Area": "Total Area"}, height=500)
                                st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.warning(f"⚠️ ไม่พบสารใดๆ ที่มีกลิ่น '{search_flavor}' ในฐานข้อมูล")
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}")
        else:
            st.error("❌ ไม่สามารถโหลดฐานข้อมูลกลิ่นได้ - ฟีเจอร์นี้ไม่สามารถใช้งานได้")

    with tabs[4]:
        st.header("📊 เปรียบเทียบโครมาโทแกรม")
        y_option = st.selectbox("เลือกค่าแกน Y:", ("Height", "Area"), key="overlay_y")
        fig_overlay = go.Figure()
        samples = combined_df['Sample'].unique()
        colors = px.colors.qualitative.Plotly
        for idx, sample in enumerate(samples):
            sample_data = combined_df[combined_df['Sample'] == sample]
            
            hovertemplate = (
                f'<b>{sample}</b>  '
                f'RT: %{{x}}  '
                f'{y_option}: %{{y}}<extra></extra>'
            )
            fig_overlay.add_trace(go.Scatter(
                x=sample_data['R.T.'], 
                y=sample_data[y_option],
                mode='lines', 
                name=sample,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=hovertemplate
            ))
        
        fig_overlay.update_layout(xaxis_title="Retention Time (นาที)", yaxis_title=y_option, legend_title_text='Samples', hovermode='x unified', height=600)
        st.plotly_chart(fig_overlay, use_container_width=True)
        st.caption("📖 **วิธีอ่าน Chromatogram Overlay:**\n- **แกน X:** Retention Time\n- **แกน Y:** ปริมาณสาร\n- Peak ที่ RT เดียวกัน = สารเดียวกัน")

    with tabs[5]:
        st.header("📄 ข้อมูลดิบทั้งหมด")
        st.dataframe(combined_df, use_container_width=True, height=600)
        st.markdown("---")
        st.download_button("📥 Download All Data (XLSX)", to_excel(combined_df), "all_combined_data.xlsx", help="ดาวน์โหลดข้อมูลดิบทั้งหมดในรูปแบบ Excel")

# ==============================================================================
# Footer
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>GC-MS Data Comparator & Flavor Explorer v2.1</p>
    <p>Powered by Streamlit | Created by Aniwat Kaewkrod</p>
</div>
""", unsafe_allow_html=True)
