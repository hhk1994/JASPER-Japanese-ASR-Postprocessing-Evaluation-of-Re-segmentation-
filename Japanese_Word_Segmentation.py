#!/usr/bin/env /users/pseudo_user_name/env/miniconda3/bin/python
import MeCab
import gzip
import os
import re
import argparse
import sys
import io
import re
from sudachipy import tokenizer
from sudachipy import dictionary
import unicodedata
import re
from clean_jpj import clean_japanese_text
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, help="Tokenization mode: char | sudachi_A | sudachi_B | sudachi_C")
args = parser.parse_args()
# Set the splitting mode (e.g., Mode.A for fine-grained segmentation)
mode_code = args.mode

# Ensure the necessary paths are included
sys.path.append('/mnt/users/pseudo_user_name/Projects/CTC/git/tools/utils/')
# Initialize the tokenizer
tokenizer_obj = dictionary.Dictionary(dict="core").create()
mode = tokenizer.Tokenizer.SplitMode.mode_code
# Define the path to the dictionary file
dct_file_name = "/mnt/users/pseudo_user_name/Projects/CTC/jpn-JPN/prepare/lm/pronunciations.dct"

# =========================
# Text Processing Utilities
# =========================
def clean_and_filter_text(text):
    """
    Apply a series of text cleaning and filtering steps:
    - Remove invalid characters (keep common Unicode ranges and punctuation)
    - Remove non-printable characters
    - Filter by Unicode category (letters, punctuation, numbers, spaces, math, currency)
    - Remove private use area Unicode characters
    - Remove extra/redundant spaces
    Returns the cleaned and filtered text.
    """
    # Remove invalid characters
    text = re.sub(r'[^\w\s.,!?;:"\'“”‘’\(\)\[\]{}<>/@#\$%&\*\+\-=\^~|`\\]', '', text)
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    # Filter by Unicode category
    def isvalid(cata):
        return cata[0] in ['L', 'P', 'N'] or cata in {'Zs', 'Sm', 'Sc'}
    text = ''.join(char for char in text if isvalid(unicodedata.category(char)))
    # Remove private use area Unicode characters
    text = re.sub(r'[\x00-\x1F\x7F\u2028\u2029\uE000-\uF8FF]', '', text)
    # Remove extra/redundant spaces
    text = re.sub(r'(?<=[\u4e00-\u9fa5\u3040-\u30ff])\s+(?=[\u4e00-\u9fa5\u3040-\u30ff])', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_only_english_letters(s):
    return bool(re.fullmatch(r'[A-Za-z]+', s))

def is_katakana_only(string):
    katakana_pattern = r'^[\u30A0-\u30FFー]+$'
    return bool(re.match(katakana_pattern, string))
    
def process_brackets(text, minlen=12):
    """
    Remove brackets and their content if the content is shorter than minlen.
    Otherwise, remove the brackets but keep the content.
    """
    def repl(match):
        content = match.group(2)
        return f' {content} ' if len(content) >= minlen else ''
    # Handle both full-width and half-width brackets
    result = re.sub(r'(\(|（)(.*?)(\)|）)', repl, text)
    return re.sub(r'\s+', ' ', result).strip()

def convert_fullwidth_to_halfwidth(text):
    """
    Convert all full-width characters in the input string to their half-width equivalents.
    """
    return ''.join(
        chr(0x0020) if ord(c) == 0x3000 else
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else
        c
        for c in text
    )

def insert_spaces_kanji_digits(text):
    """
    Insert spaces between Kanji digits and common suffixes for better tokenization.
    """
    kanji = r'[〇零一二三四五六七八九十百千万億兆]'
    suffix = r"(日|番|丁|月|時|度|目|年|条|世紀|分|号|巻|便|軒|件|チャンネル|メートル|キロ|センチ|円|秒|パーセント|メーター)"
    text = re.sub(rf'(?={kanji}+{suffix})', ' ', text)
    text = re.sub(f'(?<={kanji})(?={kanji})', ' ', text)
    text = re.sub(f'(?<={kanji})(?={suffix})', ' ', text)
    return text

def arabic_to_chinese_numeral(num):
    """
    Convert an integer (Arabic numeral) to its Chinese numeral representation.
    """
    if not (0 <= num < 10**8):
        try:
            num = int(str(num)[:8])
        except Exception:
            return str(num)[:8]
    units = ['', '十', '百', '千', '万', '十', '百', '千', '亿']
    digits = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    if num == 0:
        return '〇'
    result = ''
    unit_position = 0
    zero_flag = False
    while num > 0:
        digit = num % 10
        if digit != 0:
            result = digits[digit] + units[unit_position] + result
            zero_flag = False
        elif not zero_flag:
            zero_flag = True
        num //= 10
        unit_position += 1
    if result.startswith('〇'):
        result = result[1:]
    if result.startswith('一十'):
        result = result[1:]
    return result

def normalize_and_convert_numbers(text):
    """
    Clean, normalize, and convert Arabic numerals in the text to Chinese numerals,
    and insert spaces between Kanji digits and common suffixes for better tokenization.
    """
    text = clean_and_filter_text(clean_japanese_text(text))
    text = process_brackets(text)
    # Convert Arabic numerals to Chinese numerals
    numpattern = re.compile(r'[0-9]+')
    for match in numpattern.findall(text):
        text = text.replace(match, arabic_to_chinese_numeral(int(match)))
    # Insert spaces for better tokenization
    text = insert_spaces_kanji_digits(text)
    # Final formatting
    text = text.replace(" ", "|").replace("_", "|").replace("−", "-")
    text = text.replace("!", " ! ").replace("〇", " 〇 ")
    return text

#####################-------End Text Processing Utilities-------#####################

# =========================
# File I/O Utilities
# =========================
def yield_lines_from_file(file_path):
    """
    Generator to read a file line by line, decoding as UTF-8.
    Supports both .gz and .txt files.
    Yields None for lines that cannot be decoded or if the file cannot be opened.
    """
    try:
        # Check the file extension to determine how to open it
        if file_path.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open
        # Open the file using the appropriate function
        with open_func(file_path, 'rb') as f:
            line_number = 0
            while True:
                try:
                    # Read one line at a time
                    line = f.readline()
                    if not line:  # End of file
                        break
                    line_number += 1
                    # Decode the line as UTF-8
                    yield line.decode('utf-8')  # Yield the decoded line
                except UnicodeDecodeError as e:
                    yield None
    except (OSError, IOError) as e:
        yield None

def open_text_file(fn, op='rt'):
    """
    Open a text or gzipped file with the specified mode and UTF-8 encoding.
    Returns a file object.
    """
    try:
        if(fn.endswith(".gz")):
            file = gzip.open(fn, op, encoding='utf-8', errors='strict')
        else:
            file = open(fn, op)
        return file
    except Exception as e:
        if not file.closed:
            file.close()
        raise e

#####################---------End File I/O Utilities-------#####################


# =========================
# Dictionary Utilities
# =========================

def find_keys_ignorcase(target, dictionary):
    target_lower = target.lower()
    return [key for key in dictionary if target_lower in str(key).lower()]

def load_dct(dct_file_name):
    """
    Load pronunciation dictionaries from a file.
    - dct: maps surface forms to a list of pronunciations (case-sensitive)
    - dct2: maps lowercased surface forms to a list of pronunciations (case-insensitive)
    Only lines after '[Data]' are processed. Each valid line should have three fields separated by '\\\\'.
    """
    dct = {}
    dct2 = {}
    body = False
    with open(dct_file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("[Data]"):
                body = True
                continue
            if not body or not line:
                continue
            # Split line into tokens, expecting format: surface\\...\\pronunciation
            token = line.split(' ')[0].split('\\\\')
            if len(token) == 3 and token[0]:
                txt, _, pron = token
                # Add to case-sensitive dictionary
                dct.setdefault(txt, []).append(pron)
                # Add to case-insensitive dictionary
                txt_lower = txt.lower()
                dct2.setdefault(txt_lower, []).append(pron)
    return dct, dct2

#####################--------End Dictionary Utilities-------#####################

# =========================
# Tokenization Utilities
# =========================

def tokenize_and_map_pronunciation(text, dct, dct2):
    """
    Tokenize Japanese text and map each token to its pronunciation using SudachiPy and custom dictionaries.
    - Tokenizes the input text using SudachiPy.
    - For each token:
        * If the token's reading is Katakana and not OOV, use that reading.
        * Otherwise, look up the pronunciation in the provided dictionaries (dct, dct2).
        * If not found, use the token itself.
    - Special handling for consecutive English tokens that appear together in the original text.
    - Returns a string with each processed token (and pronunciation if available) on a new line.
    """
    text = text.strip()[:16380]
    orig_text = text
    words = tokenizer_obj.tokenize(text, mode)
    texts = []
    last_txt = ""
    
    for word in words:
        txt = word.surface().replace(" ", "_").replace("\t", "_")
        if not txt:
            continue
        prons = word.reading_form()
        # Prefer Katakana reading if available and not OOV
        if prons and is_katakana_only(prons) and not word.is_oov():
            newtxt = f'{txt}\\\\{prons}'
        else:
            # Try dictionary lookup (case-sensitive, then lowercased)
            pron = dct.get(txt, [None])[0] or dct2.get(txt.lower(), [None])[0]
            newtxt = f'{txt}\\\\{pron}' if pron else txt
        # Merge consecutive English tokens if they appear together in the original text
        if (last_txt and is_only_english_letters(last_txt) and is_only_english_letters(txt)
                and (last_txt + txt) in orig_text):
            mergetxt = texts[-1].split("\\\\")[0] + txt
            pron = dct.get(mergetxt, [None])[0]
            texts[-1] = f'{mergetxt}\\\\{pron}' if pron else mergetxt
        else:
            texts.append(newtxt[:100])
        last_txt = txt
    # Filter out tokens containing '|'
    return '\n'.join(x for x in texts if "|" not in x)

def process_text_block_to_file(texts, dct,dct2, out):
    if(len(texts) > 0):
        context = ' '.join(texts)
        context = clean_and_filter_text(context)
        try:
            context=convert_fullwidth_to_halfwidth(normalize_and_convert_numbers(context))
        except:
            pass
        context = tokenize_and_map_pronunciation(context, dct,dct2)
        out.write(f'NEW_DOCUMENT\n{context}\n')

def process_line_to_file(text, dct,dct2, out):
    if(len(text) > 0):
        context = text.strip()
        context = clean_and_filter_text(context)
        try:
            context=convert_fullwidth_to_halfwidth(normalize_and_convert_numbers(context))
        except:
            pass
        context = tokenize_and_map_pronunciation(context, dct,dct2)
        out.write(f'NEW_DOCUMENT\n{context}\n')

#####################--------End Tokenization Utilities--------#####################


def process_files(inputs, dct, dct2, outputs=None, mode='auto'):
    """
    Flexible file processing utility.
    - If `inputs` is a list file, processes each pair in the list.
    - If `outputs` is provided, writes to that file; otherwise, generates output paths.
    - `mode` can be 'auto', 'block', or 'line':
        * 'auto': try raw line processing first, fallback to block processing.
        * 'block': process as block-based (NEW_DOCUMENT separated).
        * 'line': process as raw line-based.
    """
    def process_block(infn, outfn):
        print(f'Processing as block: {infn} -> {outfn}')
        os.makedirs(os.path.dirname(outfn), exist_ok=True)
        texts = []
        with open_text_file(outfn, 'wt') as out:
            for line in yield_lines_from_file(infn):
                if line is None:
                    continue
                if line.startswith('NEW_DOCUMENT'):
                    process_text_block_to_file(texts, dct, dct2, out)
                    texts = []
                else:
                    text = line.strip().split('\\')[0]
                    texts.append(text)
            process_text_block_to_file(texts, dct, dct2, out)
        return outfn

    def process_line(infn, outfn):
        print(f'Processing as line: {infn} -> {outfn}')
        os.makedirs(os.path.dirname(outfn), exist_ok=True)
        with open_text_file(outfn, 'wt') as out:
            for line in yield_lines_from_file(infn):
                if line is None:
                    continue
                text = clean_and_filter_text(line.strip())
                if text:
                    process_line_to_file(text, dct, dct2, out)
        return outfn

    def get_outfn(infn):
        sp = infn.split('/')
        anchor = sp.index('file_path_name_pattern') if 'file_path_name_pattern' in sp else 0
        outfn = '/'.join(sp[anchor:])
        current_path = os.getcwd()
        return os.path.join(current_path, outfn)

    # If inputs is a list file, process each pair
    if os.path.isfile(inputs) and inputs.endswith('.list'):
        print(f'Processing list file: {inputs}')
        items = [x.strip() for x in open(inputs)]
        outlist = []
        outlist_path = inputs.replace('list/', 'list_processed/')
        os.makedirs(os.path.dirname(outlist_path), exist_ok=True)
        with open(outlist_path, 'wt') as out:
            for pair in zip(items[::2], items[1::2]):
                infn = pair[0]
                outfn = get_outfn(pair[1].lstrip('==> '))
                try:
                    if mode == 'block':
                        process_block(infn, outfn)
                    elif mode == 'line':
                        process_line(infn, outfn)
                    else:  # auto
                        try:
                            process_line(infn, outfn)
                        except Exception:
                            process_block(infn, outfn)
                except Exception as e:
                    print(f"Error processing {infn}: {e}")
                out.write(f'{infn}\n==> {outfn}\n')
        return outlist_path
    # Otherwise, process a single file
    else:
        infn = inputs
        outfn = outputs or get_outfn(infn)
        try:
            if mode == 'block':
                return process_block(infn, outfn)
            elif mode == 'line':
                return process_line(infn, outfn)
            else:  # auto
                try:
                    return process_line(infn, outfn)
                except Exception:
                    return process_block(infn, outfn)
        except Exception as e:
            print(f"Error processing {infn}: {e}")
            return None

def main():
    """
    Unified entry point for processing files or lists of files.
    - Use -i for input file or list file.
    - Use -o for output file (optional, for single file processing).
    - Use --mode to specify 'auto', 'block', or 'line' processing.
    """
    parser = argparse.ArgumentParser(description='Convert master to nemo json format.')
    parser.add_argument('-i', type=str, required=True, dest='input_file', help='Input file or list file')
    parser.add_argument('-o', type=str, default=None, dest='output_file', help='Output file (optional)')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'block', 'line'], help='Processing mode')
    opts = parser.parse_args()

    print(f"Loading dictionary from {dct_file_name}")
    dct, dct2 = load_dct(dct_file_name)

    result = process_files(
        inputs=opts.input_file,
        dct=dct,
        dct2=dct2,
        outputs=opts.output_file,
        mode=opts.mode
    )
    print(f"Processing complete. Output: {result}")

if __name__ == "__main__":
    print(f"{sys.argv}")
    main()
