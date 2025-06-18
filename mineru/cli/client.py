# Copyright (c) Opendatalab. All rights reserved.
import os
import click
from pathlib import Path
from loguru import logger
import sys
import logging

from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
from mineru.utils.user_display import ProgressDisplayer
from ..version import __version__
from .common import do_parse, read_fn, pdf_suffixes, image_suffixes

@click.command()
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
@click.option(
    '-p',
    '--path',
    'input_path',
    type=click.Path(exists=True),
    required=True,
    help='local filepath or directory. support pdf, png, jpg, jpeg files',
)
@click.option(
    '-o',
    '--output',
    'output_dir',
    type=click.Path(),
    required=True,
    help='output local directory',
)
@click.option(
    '-m',
    '--method',
    'method',
    type=click.Choice(['auto', 'txt', 'ocr']),
    help="""the method for parsing pdf:
    auto: Automatically determine the method based on the file type.
    txt: Use text extraction method.
    ocr: Use OCR method for image-based PDFs.
    Without method specified, 'auto' will be used by default.
    Adapted only for the case where the backend is set to "pipeline".""",
    default='auto',
)
@click.option(
    '-b',
    '--backend',
    'backend',
    type=click.Choice(['pipeline', 'vlm-transformers', 'vlm-sglang-engine', 'vlm-sglang-client']),
    help="""the backend for parsing pdf:
    pipeline: More general.
    vlm-transformers: More general.
    vlm-sglang-engine: Faster(engine).
    vlm-sglang-client: Faster(client).
    without method specified, pipeline will be used by default.""",
    default='pipeline',
)
@click.option(
    '-l',
    '--lang',
    'lang',
    type=click.Choice(['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']),
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
    Without languages specified, 'ch' will be used by default.
    Adapted only for the case where the backend is set to "pipeline".
    """,
    default='ch',
)
@click.option(
    '-u',
    '--url',
    'server_url',
    type=str,
    help="""
    When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """,
    default=None,
)
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='The starting page for PDF parsing, beginning from 0.',
    default=0,
)
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='The ending page for PDF parsing, beginning from 0.',
    default=None,
)
@click.option(
    '-f',
    '--formula',
    'formula_enable',
    type=bool,
    help='Enable formula parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-t',
    '--table',
    'table_enable',
    type=bool,
    help='Enable table parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-d',
    '--device',
    'device_mode',
    type=str,
    help='Device mode for model inference, e.g., "cpu", "cuda", "cuda:0", "npu", "npu:0", "mps". Adapted only for the case where the backend is set to "pipeline". ',
    default=None,
)
@click.option(
    '--vram',
    'virtual_vram',
    type=int,
    help='Upper limit of GPU memory occupied by a single process. Adapted only for the case where the backend is set to "pipeline". ',
    default=None,
)
@click.option(
    '--source',
    'model_source',
    type=click.Choice(['huggingface', 'modelscope', 'local']),
    help="""
    The source of the model repository. Default is 'huggingface'.
    """,
    default='huggingface',
)
@click.option(
    '--show_progress',
    'show_progress',
    type=bool,
    help='Enable tqdm progress bar display. Default is False.',
    default=False,
)
@click.option(
    '--log_level',
    'log_level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    help='Set the logging level. Default is INFO.',
    default='INFO',
)
@click.option(
    '--user_friendly_progress',
    'user_friendly_progress',
    is_flag=True,
    default=False,
    help='启用为最终用户设计的、简洁易懂的中文进度显示。',
)

def main(input_path, output_dir, method, backend, lang, server_url, start_page_id, end_page_id, formula_enable, table_enable, device_mode, virtual_vram, model_source, show_progress, log_level, user_friendly_progress):
    # 设置loguru日志级别
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    # 设置标准库logging日志级别
    logging.basicConfig(level=getattr(logging, log_level))
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(getattr(logging, log_level))

    # 初始化用户友好的进度显示器
    displayer = ProgressDisplayer(enabled=user_friendly_progress)

    if not backend.endswith('-client'):
        def get_device_mode() -> str:
            if device_mode is not None:
                return device_mode
            else:
                return get_device()
        if os.getenv('MINERU_DEVICE_MODE', None) is None:
            os.environ['MINERU_DEVICE_MODE'] = get_device_mode()

        def get_virtual_vram_size() -> int:
            if virtual_vram is not None:
                return virtual_vram
            if get_device_mode().startswith("cuda") or get_device_mode().startswith("npu"):
                return round(get_vram(get_device_mode()))
            return 1
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) is None:
            os.environ['MINERU_VIRTUAL_VRAM_SIZE']= str(get_virtual_vram_size())

        if os.getenv('MINERU_MODEL_SOURCE', None) is None:
            os.environ['MINERU_MODEL_SOURCE'] = model_source

    os.makedirs(output_dir, exist_ok=True)

    def parse_doc(path_list: list[Path]):
        try:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            total_pages = 0
            
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
                
                # 计算总页数
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(pdf_bytes)
                total_pages += len(pdf)
                pdf.close()
            
            # 显示开始处理信息
            displayer.show(f"收到任务，开始分析这份共 {total_pages} 页的文档...", is_major_step=True)
            
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                p_formula_enable=formula_enable,
                p_table_enable=table_enable,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                show_progress=show_progress,
                displayer=displayer
            )
        except Exception as e:
            logger.exception(e)

    if os.path.isdir(input_path):
        doc_path_list = []
        for doc_path in Path(input_path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)
        parse_doc(doc_path_list)
    else:
        parse_doc([Path(input_path)])

if __name__ == '__main__':
    main()
