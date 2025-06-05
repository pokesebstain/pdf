import os
import logging
import tempfile
from datetime import datetime
import streamlit as st
import tabula
import pandas as pd
import base64
import pdfplumber

# 简化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MAX_FILE_SIZE = 32 * 1024 * 1024  # 32MB max file size

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PDFProcessor:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.temp_dir = tempfile.mkdtemp(dir=upload_folder)
        logger.info(f"Created temp directory: {self.temp_dir}")

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning temp directory: {e}")

    def _clean_column_name(self, col, idx):
        """统一列名清理逻辑，包含常见列名修正"""
        if pd.isna(col) or col == '':
            return f'列{idx+1}'
        
        # 转换为字符串并清理首尾空格
        col_name = str(col).strip()
        
        # 替换换行符为空格
        col_name = col_name.replace('\n', ' ').replace('\r', ' ')
        
        # 清理连续的空格
        col_name = ' '.join(col_name.split())
        
        # 常见列名修正字典
        common_columns = {
            'epartmen': 'department',
            'ept': 'department',
            'dept': 'department',
            'nam': 'name',
            'nme': 'name',
            'id': 'ID',
            'num': 'number',
            'no': 'number',
            'desc': 'description',
            'qty': 'quantity',
            'amt': 'amount'
        }
        
        # 检查是否需要修正
        col_name_lower = col_name.lower()
        for wrong, correct in common_columns.items():
            if col_name_lower == wrong or col_name_lower.startswith(wrong):
                logger.info(f"列名修正: {col_name} -> {correct}")
                return correct
        
        # 如果清理后为空，使用默认列名
        return col_name or f'列{idx+1}'

    def _calculate_similarity(self, table1, table2):
        """计算两个表格的相似度"""
        try:
            # 确保两个表格具有相同的列
            common_cols = set(table1.columns) & set(table2.columns)
            if not common_cols:
                return 0.0

            # 只比较共同的列
            t1 = table1[list(common_cols)].fillna('')
            t2 = table2[list(common_cols)].fillna('')

            # 将表格转换为字符串数组
            arr1 = t1.values.flatten()
            arr2 = t2.values.flatten()

            # 计算相同元素的比例
            matches = sum(1 for a, b in zip(arr1, arr2) if str(a).strip() == str(b).strip())
            total = len(arr1)

            return matches / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"计算表格相似度时出错: {str(e)}")
            return 0.0

    def _is_duplicate_table(self, new_table, existing_tables):
        """检查表格是否是重复的"""
        for existing_table in existing_tables:
            # 检查维度是否相同
            if new_table.shape == existing_table.shape:
                # 计算相似度
                similarity = self._calculate_similarity(
                    new_table.drop(['来源文件', '表格序号'], axis=1, errors='ignore'),
                    existing_table.drop(['来源文件', '表格序号'], axis=1, errors='ignore')
                )
                if similarity > 0.8:  # 80%相似度认为是重复
                    logger.debug(f"发现重复表格，相似度: {similarity:.2%}")
                    return True
        return False

    def _try_extract(self, pdf_path, header_opt):
        """尝试使用指定header参数提取表格"""
        try:
            logger.info(f"开始提取表格: {os.path.basename(pdf_path)}, header={header_opt}")
            
            # 使用lattice模式提取表格，添加更多参数以提高识别准确性
            params = {
                'lattice': True,
                'stream': False,
                'guess': True,  # 启用自动检测
                'columns': None,  # 自动检测列
                'pages': 'all',
                'multiple_tables': True,
                'relative_area': True,  # 使用相对区域
            }
            
            valid_tables = []
            
            try:
                tables = tabula.read_pdf(
                    pdf_path,
                    pandas_options={
                        'header': header_opt
                    },
                    java_options=['-Dfile.encoding=UTF8', '-Xmx512m'],
                    encoding='utf-8',
                    silent=False,
                    **params
                )
                
                if tables:
                    logger.debug(f"提取到{len(tables)}个原始表格")
                    for table in tables:
                        if table is None or table.empty:
                            continue
                            
                        if len(table.index) >= 1 and len(table.columns) >= 2:  # 降低最小列数要求到2列
                            # 清理表格数据
                            table = (table
                                .replace({r'\r': '', r'\n': ' '}, regex=True)
                                .fillna('')
                                .dropna(how='all', axis=0)
                                .dropna(how='all', axis=1))
                            
                            if not table.empty:
                                # 添加来源信息
                                table['来源文件'] = os.path.basename(pdf_path)
                                table['表格序号'] = f'Table_{len(valid_tables) + 1}'
                                
                                # 检查是否重复表格
                                if not self._is_duplicate_table(table, valid_tables):
                                    valid_tables.append(table)
                                    logger.info(f"找到新的有效表格，当前共有{len(valid_tables)}个表格")
                            
            except Exception as e:
                logger.warning(f"表格提取失败: {str(e)}")
            
            # 如果lattice模式失败，尝试stream模式，同样使用优化的参数
            if not valid_tables:
                logger.info("lattice模式未提取到表格，尝试stream模式")
                params['lattice'] = False
                params['stream'] = True
                params['guess'] = True
                params['columns'] = None  # 自动检测列
                
                try:
                    tables = tabula.read_pdf(
                        pdf_path,
                        pandas_options={
                            'header': header_opt
                        },
                        java_options=['-Dfile.encoding=UTF8', '-Xmx512m'],
                        encoding='utf-8',
                        silent=False,
                        **params
                    )
                    
                    if tables:
                        for table in tables:
                            if table is None or table.empty:
                                continue
                                
                            if len(table.index) >= 1 and len(table.columns) >= 5:
                                table = (table
                                    .replace({r'\r': '', r'\n': ' '}, regex=True)
                                    .fillna('')
                                    .dropna(how='all', axis=0)
                                    .dropna(how='all', axis=1))
                                
                                if not table.empty:
                                    table['来源文件'] = os.path.basename(pdf_path)
                                    table['表格序号'] = f'Table_{len(valid_tables) + 1}'
                                    
                                    if not self._is_duplicate_table(table, valid_tables):
                                        valid_tables.append(table)
                                        logger.info(f"使用stream模式找到新的有效表格，当前共有{len(valid_tables)}个表格")
                                        
                except Exception as e:
                    logger.warning(f"stream模式提取失败: {str(e)}")
            
            logger.info(f"最终提取到 {len(valid_tables)} 个有效表格")
            return valid_tables
            
        except Exception as e:
            logger.error(f"表格提取过程出错 (header={header_opt}): {e}")
            return []

    def extract_tables_from_pdf(self, pdf_path):
        """从PDF文件中提取表格，自动尝试不同header参数"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"文件不存在: {pdf_path}")
                return []
                
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                logger.error(f"空文件: {pdf_path}")
                return []
                
            logger.info(f"开始处理文件: {os.path.basename(pdf_path)}, 大小: {file_size/1024:.1f}KB")

            all_tables = self._try_extract(pdf_path, 0)
            if not all_tables:
                logger.info(f"header=0未提取到有效表格，尝试header=None: {os.path.basename(pdf_path)}")
                all_tables = self._try_extract(pdf_path, None)

            processed_tables = []
            
            if all_tables:
                for table_index, table in enumerate(all_tables):
                    try:
                        if table is None or table.empty:
                            continue
                            
                        # 清理列名
                        new_columns = []
                        seen_columns = set()
                        for idx, col in enumerate(table.columns):
                            col_name = self._clean_column_name(col, idx)
                            
                            base_name = col_name
                            counter = 1
                            while col_name in seen_columns:
                                col_name = f"{base_name}_{counter}"
                                counter += 1
                            seen_columns.add(col_name)
                            new_columns.append(col_name)
                            
                        table.columns = new_columns
                        processed_tables.append(table)
                        
                    except Exception as e:
                        logger.error(f"处理表格{table_index+1}时出错: {str(e)}")
                        continue
                        
            if not processed_tables:
                logger.warning(f"未从文件中提取到有效表格: {os.path.basename(pdf_path)}")
                    
            logger.info(f"成功从文件提取{len(processed_tables)}个表格: {os.path.basename(pdf_path)}")
            return processed_tables
            
        except Exception as e:
            logger.error(f"提取PDF表格时发生错误: {str(e)}")
            return []

    def merge_tables(self, tables_data):
        """合并所有表格到一个sheet中，保持来源文件和表格序号列"""
        if not tables_data:
            return None

        try:
            # 收集所有表格
            all_tables = []
            for tables in tables_data:
                all_tables.extend(tables)

            if not all_tables:
                logger.warning("No tables to merge")
                return None

            # 创建一个空的DataFrame来存储所有合并的表格
            merged_data = []
            
            for table in all_tables:
                # 清理数据
                table = table.replace({r'\r': '', r'\n': ' '}, regex=True)
                table = table.fillna('')
                
                # 确保来源文件和表格序号列在最前面
                cols = list(table.columns)
                source_cols = ['来源文件', '表格序号']
                other_cols = [col for col in cols if col not in source_cols]
                table = table[source_cols + other_cols]
                
                # 添加当前表格
                merged_data.append(table)
            
            # 垂直合并所有数据
            final_df = pd.concat(merged_data, ignore_index=True)
            
            # 创建Excel文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.temp_dir, f'提取的表格_{timestamp}.xlsx')

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 写入合并后的数据
                final_df.to_excel(writer, sheet_name='提取的表格', index=False)
                
                # 获取工作表对象
                worksheet = writer.sheets['提取的表格']
                
                # 设置列宽
                for idx, col in enumerate(final_df.columns):
                    max_length = max(
                        final_df[col].astype(str).apply(len).max(),
                        len(str(col))
                    )
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

            logger.info(f"Successfully saved {len(all_tables)} tables into {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error merging tables: {e}")
            return None

    def process_pdfs(self, files):
        """处理多个PDF文件并将表格合并到一个Excel文件中"""
        if not files:
            return None, []

        try:
            all_tables = []
            failed_files = []

            if not isinstance(files, (list, tuple)):
                logger.error(f"无效的输入类型: {type(files)}")
                return None, [("input", f"无效的输入类型: {type(files)}")]

            for file in files:
                if not hasattr(file, 'read'):
                    logger.error(f"无效的文件对象类型: {type(file)}")
                    failed_files.append((str(type(file)), "无效的文件对象"))
                    continue

                filename = getattr(file, 'name', getattr(file, 'filename', None))
                logger.debug(f"原始文件名: {filename} (类型: {type(filename)})")
                
                if filename is None:
                    logger.error("无法获取文件名")
                    failed_files.append(("unknown", "无法获取文件名"))
                    continue
                    
                try:
                    filename = str(filename)
                    if not isinstance(filename, str):
                        raise TypeError(f"文件名转换失败，得到类型: {type(filename)}")
                except Exception as e:
                    logger.error(f"文件名转换错误: {str(e)}")
                    failed_files.append((str(filename), f"文件名转换错误: {str(e)}"))
                    continue
                    
                if not filename.lower().endswith('.pdf'):
                    logger.warning(f"跳过非PDF文件: {filename}")
                    continue

                try:
                    pdf_path = os.path.join(self.temp_dir, filename)
                    logger.info(f"构造的路径: {pdf_path} (类型: {type(pdf_path)})")
                    
                    if not isinstance(pdf_path, (str, bytes, os.PathLike)):
                        raise TypeError(f"无效的文件路径类型: {type(pdf_path)}")
                        
                    with open(pdf_path, 'wb') as f:
                        file_content = file.read()  # 使用read()而不是getvalue()
                        if not file_content:
                            logger.warning(f"空文件内容: {filename}")
                            failed_files.append((filename, "空文件内容"))
                            continue
                        f.write(file_content)

                    if not os.path.exists(pdf_path):
                        logger.error(f"文件保存失败: {filename}")
                        failed_files.append((filename, "文件保存失败"))
                        continue

                    file_size = os.path.getsize(pdf_path)
                    if file_size == 0:
                        logger.warning(f"空文件: {filename}")
                        failed_files.append((filename, "文件为空"))
                        continue

                    tables = self.extract_tables_from_pdf(pdf_path)
                    if tables:
                        all_tables.append(tables)
                        logger.info(f"成功从{filename}提取{len(tables)}个表格")
                    else:
                        logger.warning(f"未从{filename}提取到表格")
                        failed_files.append((filename, "未提取到表格"))

                except Exception as e:
                    logger.error(f"处理文件失败: {filename}, 错误: {str(e)}")
                    failed_files.append((filename, f"处理失败: {str(e)}"))
                    continue

            if all_tables:
                output_path = self.merge_tables(all_tables)
                if output_path:
                    logger.info(f"成功合并表格到: {output_path}")
                    return output_path, failed_files
                else:
                    logger.error("合并表格失败")
                    return None, failed_files + [("merge", "合并表格失败")]
            else:
                logger.warning("没有可合并的表格")
                return None, failed_files

        except Exception as e:
            logger.error(f"处理PDF文件时发生错误: {str(e)}")
            return None, [("process", f"处理失败: {str(e)}")]

def get_download_link(file_path, link_text="下载Excel文件"):
    """为文件创建一个下载链接"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        filename = os.path.basename(file_path)
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        logger.error(f"创建下载链接失败: {e}")
        return None

def main():
    st.set_page_config(
        page_title="PDF表格提取工具",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("PDF表格提取工具")
    st.markdown("""
    ### 将PDF文件中的表格提取到Excel
    
    上传一个或多个PDF文件，自动提取其中的表格并保存到Excel文件中。所有表格将按顺序保存在同一个sheet中。
    
    **支持功能**:
    - 批量处理多个PDF文件
    - 自动识别和提取表格
    - 保留完整的列名信息
    - 来源文件和表格序号单独列示
    - 导出为Excel格式，便于查看和编辑
    """)
    
    processor = PDFProcessor(UPLOAD_FOLDER)
    
    uploaded_files = st.file_uploader("选择PDF文件", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")
        
        if st.button("开始处理"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("正在处理文件...")
                output_path, failed_files = processor.process_pdfs(uploaded_files)
                
                progress_bar.progress(100)
                
                if output_path:
                    st.success(f"处理完成! 成功提取表格并保存到Excel文件。")
                    
                    download_link = get_download_link(output_path)
                    if download_link:
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.error("创建下载链接失败")
                    
                    try:
                        preview_df = pd.read_excel(output_path)
                        st.subheader("数据预览")
                        st.dataframe(preview_df.head(20))
                    except Exception as e:
                        st.error(f"无法预览数据: {e}")
                
                if failed_files:
                    st.warning(f"有 {len(failed_files)} 个文件处理失败")
                    failure_df = pd.DataFrame(failed_files, columns=["文件", "原因"])
                    st.dataframe(failure_df)
            
            except Exception as e:
                st.error(f"处理过程中发生错误: {str(e)}")
                logger.error(f"处理过程中发生错误: {str(e)}")
            
            finally:
                try:
                    processor.cleanup()
                except Exception as e:
                    logger.error(f"清理临时文件失败: {e}")
    
    with st.expander("使用说明"):
        st.markdown("""
        #### 如何使用
        
        1. 点击"选择PDF文件"上传一个或多个PDF文件
        2. 点击"开始处理"按钮
        3. 等待处理完成
        4. 点击"下载Excel文件"获取结果
        
        #### 注意事项
        
        - 支持的最大文件大小为32MB
        - 仅支持包含表格的PDF文件
        - 处理大文件可能需要较长时间
        - 所有表格将按顺序保存在同一个sheet中
        - 来源文件和表格序号将显示在每行的前两列
        - 如果表格提取失败，请尝试使用更清晰的PDF文件
        """)
    
    st.markdown("---")
    st.markdown("PDF表格提取工具 | 版本 1.0")

if __name__ == "__main__":
    main()
