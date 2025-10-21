# app.py (เวอร์ชันสมบูรณ์ที่สุด - เพิ่ม Credit ผู้สร้าง)

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

# --- ส่วนที่ 1: ฟังก์ชัน (ไม่เปลี่ยนแปลง) ---
def clean_compound_name(name):
    """
    ทำความสะอาดชื่อสารประกอบ: ลบเครื่องหมายคำพูด, ช่องว่าง, ตัวเลข/สัญลักษณ์ที่ไม่ใช่ชื่อสาร, และปรับรูปแบบตัวพิมพ์
    """
    if not isinstance(name, str):
        return None
    cleaned_name = name.strip().strip('"')
    
    cleaned_name = re.sub(r'^\d+[\s\-\.]*', '', cleaned_name)
    cleaned_name = re.sub(r'^\([\+\-]\d?\)[\s\-\.]*', '', cleaned_name)
    cleaned_name = re.sub(r'[\s\-\.]*\(\d?\)$', '', cleaned_name)
    cleaned_name = re.sub(r'[\s\-\.]*\[\d?\].*$', '', cleaned_name)
    cleaned_name = re.sub(r'\b\d+\b', '', cleaned_name)
    cleaned_name = re.sub(r'\b[A-Z]\b', '', cleaned_name)

    cleaned_name = cleaned_name.strip()
    
    if not cleaned_name:
        return None
    
    if not re.search(r'[a-zA-Z]{3,}', cleaned_name):
        return None
    
    if cleaned_name.isupper():
        cleaned_name = cleaned_name.title()
    
    return cleaned_name

def parse_report_file(uploaded_file):
    """
    อ่านและประมวลผลไฟล์ Report ของ GC-MS/FID
    """
    try:
        sample_name = os.path.splitext(uploaded_file.name)[0]
        file_content = uploaded_file.getvalue().decode("utf-8", errors='ignore')
        lines = [line.strip() for line in file_content.splitlines()]

        peak_list_header_index = -1
        library_search_header_index = -1

        for i, line in enumerate(lines):
            if line.startswith('"Peak","R.T."'):
                peak_list_header_index = i
            elif line.startswith('"PK","RT"'):
                library_search_header_index = i

        if peak_list_header_index == -1:
            return None

        header1 = [name.strip('"') for name in lines[peak_list_header_index].split(',')]
        data_rows1 = []
        
        end_index1 = library_search_header_index if library_search_header_index != -1 else len(lines)
        
        for i in range(peak_list_header_index + 1, end_index1):
            parts = [p.strip() for p in lines[i].split(',')]
            if len(parts) > 1 and parts[0].isdigit():
                data_rows1.append(parts)
        
        if not data_rows1:
            return None

        df_peaks = pd.DataFrame(data_rows1)
        num_cols = min(len(df_peaks.columns), len(header1))
        df_peaks = df_peaks.iloc[:, :num_cols]
        df_peaks.columns = header1[:num_cols]

        df_library = None
        if library_search_header_index != -1:
            header2 = [name.strip('"') for name in lines[library_search_header_index].split(',')]
            data_rows2 = []
            for i in range(library_search_header_index + 1, len(lines)):
                parts = [p.strip() for p in lines[i].split(',')]
                if len(parts) > 1 and parts[0].isdigit():
                    data_rows2.append(parts)
            
            if data_rows2:
                df_library = pd.DataFrame(data_rows2)
                num_cols_lib = min(len(df_library.columns), len(header2))
                df_library = df_library.iloc[:, :num_cols_lib]
                df_library.columns = header2[:num_cols_lib]

        cols_to_convert = ["Peak", "R.T.", "Height", "Area", "Pct Max", "Pct Total"]
        for col in cols_to_convert:
            if col in df_peaks.columns:
                df_peaks[col] = pd.to_numeric(df_peaks[col], errors='coerce')
        df_peaks.dropna(subset=['Peak', 'R.T.'], inplace=True)

        if df_library is not None and 'PK' in df_library.columns:
            df_library['PK'] = pd.to_numeric(df_library['PK'], errors='coerce')
            df_merged = pd.merge(df_peaks, df_library.rename(columns={'PK': 'Peak'}), on='Peak', how='left')
            df_merged.loc[df_merged['Library/ID'].isna(), 'Library/ID'] = 'Unknown'
        else:
            df_merged = df_peaks.copy()
            df_merged['Library/ID'] = 'Unknown'
        
        df_merged['Sample'] = sample_name
        return df_merged
    except Exception:
        return None

def to_excel(df):
    """
    แปลง DataFrame เป็นไฟล์ Excel ในรูปแบบ BytesIO
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data') 
    processed_data = output.getvalue()
    return processed_data

# --- ส่วนที่ 2: การสร้าง UI และจัดการสถานะ ---
st.set_page_config(layout="wide")
st.title("Multi-Sample GC-MS Data Comparator")

# === จุดแก้ไข: เพิ่ม Credit ที่นี่ (เวอร์ชันใหม่) ===
st.markdown('<p style="color:green; font-weight:bold;">Created by Aniwat Kaewkrod</p>', unsafe_allow_html=True)
# ===============================================

st.write("อัปโหลดไฟล์ Report หลายไฟล์ (สูงสุด 20 ไฟล์) เพื่อเปรียบเทียบข้อมูล")

# การจัดการ Session State เพื่อคงสถานะของแอป
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

st.sidebar.header("File Upload")
uploaded_files = st.sidebar.file_uploader(
    "อัปโหลดไฟล์ Report ของคุณ",
    type=None,
    accept_multiple_files=True,
    key="file_uploader"
)

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
analyze_button = col1.button("🚀 Analyze", use_container_width=True, type="primary")
clear_button = col2.button("🗑️ Clear Data", use_container_width=True)

if clear_button:
    st.session_state.analysis_complete = False
    st.session_state.uploaded_files_list = []
    st.rerun()

if analyze_button and uploaded_files:
    if len(uploaded_files) > 20:
        st.error("ขออภัย, สามารถเปรียบเทียบได้สูงสุดครั้งละ 20 ไฟล์เท่านั้น")
    else:
        st.session_state.uploaded_files_list = uploaded_files
        st.session_state.analysis_complete = True
elif analyze_button and not uploaded_files:
    st.warning("กรุณาอัปโหลดไฟล์ก่อนกด Analyze")

# --- ส่วนแสดงผลหลัก ---
if not st.session_state.analysis_complete:
    st.info("กรุณาอัปโหลดไฟล์ในแถบด้านข้าง แล้วกดปุ่ม 'Analyze' เพื่อเริ่มต้น")
    if uploaded_files:
        st.write("ไฟล์ที่พร้อมสำหรับการวิเคราะห์:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

if st.session_state.analysis_complete and st.session_state.uploaded_files_list:
    
    with st.spinner("Processing data... Please wait."):
        all_data_list = [parse_report_file(f) for f in st.session_state.uploaded_files_list]
        valid_data_list = [df for df in all_data_list if df is not None]

    if not valid_data_list:
        st.error("ไม่สามารถประมวลผลไฟล์ใดๆ ได้เลย กรุณาตรวจสอบรูปแบบไฟล์")
        st.session_state.analysis_complete = False
        st.stop()

    combined_df = pd.concat(valid_data_list, ignore_index=True)
    
    combined_df['Compound'] = combined_df['Library/ID'].apply(clean_compound_name)
    base_analysis_df = combined_df.dropna(subset=['Compound'])
    base_analysis_df = base_analysis_df[base_analysis_df['Compound'] != 'Unknown']

    tab_heatmap, tab_pca, tab_overlay, tab_data = st.tabs([
        "🔥 Comparative Heatmap", 
        "🧬 PCA Clustering", 
        "📊 Overlaid Chromatograms", 
        "📄 Combined Data"
    ])

    with tab_heatmap:
        st.header("เปรียบเทียบปริมาณสารระหว่างตัวอย่าง (Heatmap)")
        st.subheader("Display Options")
        
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        with opt_col1:
            value_option = st.selectbox("เลือกค่าที่จะแสดง:", ("Area", "Height"), key="heatmap_value")
        with opt_col2:
            colorscale_option = st.selectbox(
                "เลือกชุดโทนสี:",
                ("Plasma", "Viridis", "Cividis", "Blues", "Reds", "Greens", "YlGnBu", "YlOrRd", "Inferno", "Magma"),
                index=0
            )
        with opt_col3:
            filter_contaminants = st.checkbox("กรองสารปนเปื้อน (Siloxanes)", value=True, key="filter_check")
        with opt_col4:
            use_log_scale = st.checkbox("ใช้สเกลสีแบบ Log", value=False, key="log_scale_check")

        st.markdown("---")
        st.subheader("Filter & Sort Compounds")
        search_term = st.text_input("ค้นหาสารประกอบ (Compound Search):", "").strip().lower()
        sort_option = st.selectbox(
            "เรียงลำดับสารประกอบ (Compound Sort):",
            ("Alphabetical (A-Z)", "Total Abundance (Highest First)", "Variance (Highest First)"),
            key="compound_sort_option"
        )

        CONTAMINANT_KEYWORDS = [
            'cyclosiloxane', 'cyclotrisiloxane', 'cyclopentasiloxane', 
            'cyclotetrasiloxane', 'cyclohexasiloxane', 'cyclododecasiloxane'
        ]
        contaminant_pattern = '|'.join(CONTAMINANT_KEYWORDS)
        
        heatmap_df = base_analysis_df.copy()
        if filter_contaminants:
            heatmap_df = heatmap_df[~heatmap_df['Compound'].str.contains(contaminant_pattern, case=False, regex=True, flags=re.IGNORECASE)]

        if search_term:
            heatmap_df = heatmap_df[heatmap_df['Compound'].str.lower().str.contains(search_term, na=False)]

        if not heatmap_df.empty:
            try:
                heatmap_pivot = heatmap_df.pivot_table(
                    index='Compound', columns='Sample', values=value_option,
                    aggfunc='sum'
                ).fillna(0)
                
                heatmap_pivot = heatmap_pivot[heatmap_pivot.sum(axis=1) > 0]
                
                if not heatmap_pivot.empty:
                    if sort_option == "Total Abundance (Highest First)":
                        heatmap_pivot['__sort_col__'] = heatmap_pivot.sum(axis=1)
                        heatmap_pivot = heatmap_pivot.sort_values(by='__sort_col__', ascending=False).drop(columns='__sort_col__')
                    elif sort_option == "Variance (Highest First)":
                        heatmap_pivot['__sort_col__'] = heatmap_pivot.var(axis=1)
                        heatmap_pivot = heatmap_pivot.sort_values(by='__sort_col__', ascending=False).drop(columns='__sort_col__')
                    else:
                        heatmap_pivot = heatmap_pivot.sort_index(ascending=True)

                    plot_data = heatmap_pivot.copy()
                    color_axis_label = f"{value_option}"
                    
                    if use_log_scale:
                        plot_data = np.log1p(plot_data)
                        color_axis_label = f"Log({value_option})"

                    num_compounds = len(plot_data.index)
                    graph_height = max(500, num_compounds * 22)

                    fig_heatmap = px.imshow(
                        plot_data,
                        labels=dict(x="Sample (ไฟล์)", y="Compound (สาร)", color=color_axis_label),
                        aspect="auto",
                        color_continuous_scale=colorscale_option,
                        height=graph_height
                    )
                    fig_heatmap.update_layout(
                        xaxis_side="top",
                        yaxis=dict(
                            tickmode='linear', 
                            automargin=True,
                            title_text="Compound (สาร)",
                            dtick=1
                        ),
                        xaxis=dict(
                            title_text="Sample (ไฟล์)"
                        ),
                        coloraxis_colorbar=dict(
                            title=color_axis_label
                        )
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    st.caption("""
                    **Heatmap:** แสดงปริมาณของสารแต่ละชนิด (แกน Y) ที่พบในแต่ละไฟล์ตัวอย่าง (แกน X)
                    - **สีเข้ม:** หมายถึงปริมาณสาร **น้อย**
                    - **สีอ่อน/สว่าง:** หมายถึงปริมาณสาร **มาก**
                    - **0, 50M, 100M, 500M:** คือค่าประมาณการของปริมาณสาร (M = ล้าน) ที่ใช้ในการกำหนดระดับสี
                    """)

                    st.markdown("---")
                    st.subheader("Export Options")
                    
                    export_col1, export_col2 = st.columns(2)

                    with export_col1:
                        img_bytes = fig_heatmap.to_image(format="jpeg", width=1200, height=graph_height, scale=2)
                        st.download_button(
                            label="📥 Download Heatmap as JPG",
                            data=img_bytes,
                            file_name="heatmap_comparison.jpg",
                            mime="image/jpeg"
                        )
                    
                    with export_col2:
                        excel_data = to_excel(heatmap_pivot)
                        st.download_button(
                            label="📥 Download Heatmap Data as XLSX",
                            data=excel_data,
                            file_name="heatmap_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                else:
                    st.warning("ไม่พบข้อมูลที่เพียงพอสำหรับสร้าง Heatmap หลังจากการกรอง")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสร้าง Heatmap หรือ Export: {e}")
        else:
            st.warning("ไม่พบสารที่ระบุชื่อได้ในไฟล์ที่อัปโหลดเพื่อนำมาสร้าง Heatmap")

    with tab_pca:
        st.header("การจัดกลุ่มตัวอย่าง (PCA Clustering)")
        st.info("แสดงความคล้ายคลึงกันของโปรไฟล์สารเคมีระหว่างตัวอย่างทั้งหมด ตัวอย่างที่อยู่ใกล้กันหมายถึงมีโปรไฟล์คล้ายกัน")

        pca_col1, pca_col2 = st.columns(2)
        pca_value_option = pca_col1.selectbox("เลือกค่าที่จะใช้คำนวณ:", ("Area", "Height"), key="pca_value")
        use_groups = pca_col2.toggle("กำหนดกลุ่มเพื่อใส่สี", value=False)
        
        group_map = {}
        if use_groups:
            st.write("กำหนดกลุ่มให้กับแต่ละตัวอย่าง:")
            all_samples = list(base_analysis_df['Sample'].unique())
            num_groups = st.number_input("จำนวนกลุ่ม", min_value=2, max_value=10, value=2, step=1, key="num_groups_pca")
            group_cols = st.columns(num_groups)
            
            selected_samples_in_groups = set()
            for i in range(num_groups):
                with group_cols[i]:
                    group_name = st.text_input(f"ชื่อกลุ่มที่ {i+1}", value=f"Group {i+1}", key=f"gname_pca_{i}")
                    
                    available_samples = [s for s in all_samples if s not in selected_samples_in_groups]
                    members = st.multiselect(
                        f"เลือกตัวอย่างสำหรับกลุ่ม {group_name}", 
                        options=available_samples, 
                        key=f"gmembers_pca_{i}"
                    )
                    for member in members:
                        group_map[member] = group_name
                        selected_samples_in_groups.add(member)

        if len(base_analysis_df['Sample'].unique()) < 2:
            st.warning("ต้องมีตัวอย่างอย่างน้อย 2 ไฟล์เพื่อทำการวิเคราะห์ PCA")
        else:
            try:
                pivot_df = base_analysis_df.pivot_table(index='Compound', columns='Sample', values=pca_value_option, aggfunc='sum').fillna(0)
                
                data_for_pca = pivot_df.T
                
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_pca)
                
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_data)
                
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data_for_pca.index)
                
                if use_groups and group_map:
                    pca_df['Group'] = pca_df.index.map(group_map).fillna('Ungrouped')
                else:
                    pca_df['Group'] = 'Sample'
                
                explained_variance = pca.explained_variance_ratio_

                st.subheader("PCA Scatter Plot")
                fig_pca = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Group',
                    text=pca_df.index,
                    labels={
                        'PC1': f'PC1 ({explained_variance[0]*100:.2f}%)',
                        'PC2': f'PC2 ({explained_variance[1]*100:.2f}%)'
                    },
                    title="2D PCA of Samples"
                )
                fig_pca.update_traces(textposition='top center')
                
                fig_pca.update_layout(
                    legend_title_text='Groups',
                    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey'),
                    yaxis=dict(
                        zeroline=True, 
                        zerolinewidth=1, 
                        zerolinecolor='LightGrey',
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                )
                
                st.plotly_chart(fig_pca, use_container_width=True)
                st.caption(f"""
                **คำอธิบาย PCA Plot:**
                - แต่ละจุดในกราฟคือ **ไฟล์ตัวอย่าง 1 ไฟล์**
                - **จุดที่อยู่ใกล้กัน** หมายถึงตัวอย่างเหล่านั้นมีโปรไฟล์สารเคมีโดยรวมที่ **คล้ายคลึงกัน**
                - **จุดที่อยู่ห่างกัน** หมายถึงโปรไฟล์สารเคมี **แตกต่างกัน**
                - **PC1 (Principal Component 1) และ PC2 (Principal Component 2)** คือแกนใหม่ที่ถูกสร้างขึ้นเพื่ออธิบายความแปรปรวนของข้อมูลได้มากที่สุด
                - **เปอร์เซ็นต์ในวงเล็บ:** แสดงว่าแกนนั้นๆ อธิบายความแปรปรวนของข้อมูลทั้งหมดได้กี่เปอร์เซ็นต์ (ยิ่งสูงยิ่งสำคัญ)
                """)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการคำนวณ PCA: {e}")

    with tab_overlay:
        st.header("เปรียบเทียบโปรไฟล์โครมาโทแกรม")
        y_overlay_option = st.selectbox("เลือกค่าสำหรับแกน Y:", ("Height", "Area"), key="overlay_y")
        fig_overlay = go.Figure()
        samples = combined_df['Sample'].unique()
        colors = px.colors.qualitative.Plotly
        
        for i, sample in enumerate(samples):
            sample_df = combined_df[combined_df['Sample'] == sample]
            fig_overlay.add_trace(go.Scatter(
                x=sample_df['R.T.'], 
                y=sample_df[y_overlay_option],
                mode='lines', 
                name=sample,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_overlay.update_layout(
            xaxis_title="Retention Time", 
            yaxis_title=y_overlay_option,
            plot_bgcolor='white',
            xaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgrey'),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgrey'),
            legend_title_text='Samples'
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
        st.caption("""
        **Chromatogram Overlay:** แสดงกราฟโครมาโทแกรมของทุกไฟล์ซ้อนทับกัน
        - **แกน X (Retention Time):** เวลาที่สารเคลื่อนที่ผ่านคอลัมน์
        - **แกน Y (Height/Area):** ปริมาณของสารที่ตรวจจับได้
        - **ประโยชน์:** ช่วยให้เห็นภาพรวมของการเปลี่ยนแปลงโปรไฟล์โครมาโทแกรมระหว่างตัวอย่างได้อย่างรวดเร็ว
        """)

    with tab_data:
        st.header("ข้อมูลทั้งหมดที่รวมกัน")
        st.dataframe(combined_df)
        
        st.markdown("---")
        excel_all_data = to_excel(combined_df)
        st.download_button(
            label="📥 Download All Combined Data as XLSX",
            data=excel_all_data,
            file_name="all_combined_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
