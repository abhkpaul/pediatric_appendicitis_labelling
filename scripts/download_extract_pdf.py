import pandas as pd
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import yaml
import subprocess
import shutil

# Try importing various PDF libraries with fallbacks
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDataExtractor:
    def __init__(self):
        # Get the directory of the currently executing script
        script_dir = Path(__file__).parent.resolve()

        # Hardcode the paths to be relative to the script's location
        # This will avoid issues with incorrect config files.
        self.config = {
            'data_extraction': {
                'git_repo_url': 'https://github.com/asherboucher87/data_orur_pediatric_appendicitis.git',
                'us_pdf_directory': script_dir / '../data/raw/pdfs/or_ur_pediatric_logs/ultrasound_pediatric_logs/',
                'or_pdf_directory': script_dir / '../data/raw/pdfs/or_ur_pediatric_logs/operative_pediatric_logs/',
                'us_output_csv': script_dir / '../data/processed/us_reports.csv',
                'or_output_csv': script_dir / '../data/processed/or_reports.csv',
                'consolidated_output_csv': script_dir / '../data/processed/merged_data.csv'
            }
        }
        logger.info("Using hardcoded paths to ensure correct directory structure.")

        self.us_reports_data = []
        self.or_reports_data = []

    def download_data(self):
        """Clone the git repository, removing the existing directory if it exists."""
        git_repo_url = self.config['data_extraction']['git_repo_url']
        script_dir = Path(__file__).parent.resolve()
        local_repo_path = script_dir / '../data/raw/pdfs'

        if os.path.exists(local_repo_path):
            logger.info(f"Removing existing directory at {local_repo_path} to ensure a fresh download.")
            shutil.rmtree(local_repo_path)

        logger.info(f"Cloning repository from {git_repo_url} to {local_repo_path}...")
        try:
            subprocess.run(['git', 'clone', git_repo_url, local_repo_path], check=True)
            logger.info("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        text = None

        # Method 1: PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                with fitz.open(pdf_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text() + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using PyMuPDF from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")

        # Method 2: pypdf
        if PYPDF_AVAILABLE and text is None:
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using pypdf from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"pypdf failed for {pdf_path}: {e}")

        # Method 3: pdfplumber
        if PDFPLUMBER_AVAILABLE and text is None:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    logger.debug(f"Successfully extracted text using pdfplumber from {os.path.basename(pdf_path)}")
                    return text.strip()
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path}: {e}")

        logger.error(f"All extraction methods failed for {pdf_path}")
        return None

    def extract_case_id(self, text, filename):
        """
        Extract case_id from PDF content or filename.
        Logic: Try matching 'Case ID: {case_id}' in text, else try filename pattern.
        """
        # Try extracting from text first
        case_id_match = re.search(r'Case ID[:\s]*([A-Za-z0-9\-]+)', text)
        if case_id_match:
            return case_id_match.group(1)
        # Try from filename (e.g., ped_ultrsndrpt_rptCASE-AB12CD-10043)
        filename_match = re.search(r'(CASE-[A-Z0-9]+-\d+)', filename)
        if filename_match:
            return filename_match.group(1)
        return None

    def parse_ultrasound_report(self, text, filename):
        """Parse ultrasound report text and extract structured data."""
        try:
            report_data = {
                'report_id': filename,
                'report_text': text,
                'extraction_timestamp': datetime.now().isoformat()
            }

            # Extract case_id
            report_data['case_id'] = self.extract_case_id(text, filename)

            # Extract exam date
            date_patterns = [
                r'Date of exam.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
                r'Exam Date.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})',
                r'Date.*?([A-Za-z]+\s+\d{1,2},\s+\d{4})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    report_data['exam_date'] = date_match.group(1)
                    break

            # Extract clinical information
            clinical_match = re.search(r'CLINICAL INFORMATION[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                       re.IGNORECASE | re.DOTALL)
            if clinical_match:
                report_data['clinical_information'] = clinical_match.group(1).strip()

            # Extract findings
            findings_match = re.search(r'FINDINGS[:\s]*(.*?)(?=IMPRESSION|$)', text, re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                report_data['findings'] = findings_text

                appendix_match = re.search(r'appendix.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if appendix_match:
                    report_data['appendix_diameter_mm'] = float(appendix_match.group(1))

                wall_match = re.search(r'wall.*?thick.*?(\d+\.?\d*)\s*mm', findings_text, re.IGNORECASE)
                if wall_match:
                    report_data['wall_thickness_mm'] = float(wall_match.group(1))

                report_data['has_free_fluid'] = 'free fluid' in findings_text.lower()
                report_data['has_fat_stranding'] = any(
                    term in findings_text.lower() for term in ['fat stranding', 'hyperechoic fat'])
                report_data['has_lymph_nodes'] = 'lymph nodes' in findings_text.lower()
                report_data['has_appendicolith'] = 'appendicolith' in findings_text.lower()

            # Extract impression and determine severity
            impression_match = re.search(r'IMPRESSION[:\s]*(.*?)(?=$|\n\n)', text, re.IGNORECASE | re.DOTALL)
            if impression_match:
                impression_text = impression_match.group(1).strip()
                report_data['impression'] = impression_text

                impression_lower = impression_text.lower()
                if 'normal' in impression_lower or 'unremarkable' in impression_lower:
                    report_data['severity_grade'] = 'Normal'
                elif 'simple' in impression_lower or 'uncomplicated' in impression_lower:
                    report_data['severity_grade'] = 'Simple/Uncomplicated'
                elif 'complicated' in impression_lower or 'gangrenous' in impression_lower:
                    report_data['severity_grade'] = 'Gangrenous'
                elif 'perforated' in impression_lower or 'ruptured' in impression_lower or 'abscess' in impression_lower:
                    report_data['severity_grade'] = 'Perforated'
                else:
                    report_data['severity_grade'] = 'Uncertain'

            return report_data

        except Exception as e:
            logger.error(f"Error parsing ultrasound report {filename}: {e}")
            return None

    def parse_operative_report(self, text, filename):
        """Parse operative report text and extract structured data."""
        try:
            report_data = {
                'report_id': filename,
                'report_text': text,
                'extraction_timestamp': datetime.now().isoformat()
            }

            # Extract case_id
            report_data['case_id'] = self.extract_case_id(text, filename)

            # Extract date
            date_match = re.search(r'Date[:\s]*(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
            if date_match:
                report_data['operation_date'] = date_match.group(1)

            # Extract surgeon
            surgeon_match = re.search(r'Surgeon[:\s]*(.*?)(?=\n|$)', text, re.IGNORECASE)
            if surgeon_match:
                report_data['surgeon'] = surgeon_match.group(1).strip()

            # Extract pre-operative diagnosis
            preop_match = re.search(r'Pre-operative diagnosis[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                    re.IGNORECASE | re.DOTALL)
            if preop_match:
                report_data['preop_diagnosis'] = preop_match.group(1).strip()

            # Extract post-operative diagnosis and determine severity
            postop_match = re.search(r'Post-operative diagnosis[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                     re.IGNORECASE | re.DOTALL)
            if postop_match:
                postop_text = postop_match.group(1).strip()
                report_data['postop_diagnosis'] = postop_text

                postop_lower = postop_text.lower()
                if 'normal' in postop_lower or 'unremarkable' in postop_lower:
                    report_data['severity_grade'] = 'Normal'
                elif 'simple' in postop_lower or 'uncomplicated' in postop_lower:
                    report_data['severity_grade'] = 'Simple/Uncomplicated'
                elif 'complicated' in postop_lower or 'gangrenous' in postop_lower:
                    report_data['severity_grade'] = 'Gangrenous'
                elif 'perforated' in postop_lower or 'ruptured' in postop_lower or 'abscess' in postop_lower:
                    report_data['severity_grade'] = 'Perforated'
                else:
                    report_data['severity_grade'] = 'Uncertain'

            # Extract operative findings
            findings_match = re.search(r'operative findings[:\s]*(.*?)(?=\n\n|\n[A-Z]+\s*[A-Z]*:|$)', text,
                                       re.IGNORECASE | re.DOTALL)
            if findings_match:
                findings_text = findings_match.group(1).strip()
                report_data['operative_findings'] = findings_text

                report_data['found_perforation'] = any(
                    term in findings_text.lower() for term in ['perforated', 'perforation', 'ruptured'])
                report_data['found_abscess'] = 'abscess' in findings_text.lower()
                report_data['found_gangrene'] = 'gangrenous' in findings_text.lower()
                report_data['found_contamination'] = any(
                    term in findings_text.lower() for term in ['contamination', 'purulent', 'fibrinous'])

            return report_data

        except Exception as e:
            logger.error(f"Error parsing operative report {filename}: {e}")
            return None

    def process_ultrasound_reports(self, pdf_directory):
        logger.info(f"Processing ultrasound reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []

        successful_extractions = 0

        for pdf_file in tqdm(pdf_files, desc="Extracting Ultrasound Reports"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                report_data = self.parse_ultrasound_report(text, pdf_file.stem)
                if report_data:
                    self.us_reports_data.append(report_data)
                    successful_extractions += 1
            else:
                logger.warning(f"Could not extract text from {pdf_file.name}")

        logger.info(f"Successfully processed {successful_extractions}/{len(pdf_files)} ultrasound reports")
        return self.us_reports_data

    def process_operative_reports(self, pdf_directory):
        logger.info(f"Processing operative reports from: {pdf_directory}")

        pdf_files = list(Path(pdf_directory).rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []

        successful_extractions = 0

        for pdf_file in tqdm(pdf_files, desc="Extracting Operative Reports"):
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                report_data = self.parse_operative_report(text, pdf_file.stem)
                if report_data:
                    self.or_reports_data.append(report_data)
                    successful_extractions += 1
            else:
                logger.warning(f"Could not extract text from {pdf_file.name}")

        logger.info(f"Successfully processed {successful_extractions}/{len(pdf_files)} operative reports")
        return self.or_reports_data

    def save_to_csv(self, data, output_path):
        if data is None or (hasattr(data, 'empty') and data.empty):
            logger.warning(f"No data to save to {output_path}")
            return False

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path} with {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV {output_path}: {e}")
            return False

    def generate_consolidated_dataset(self):
        """Generate the final consolidated dataset by merging US and OR reports using case_id (not patient_id)."""
        if not self.us_reports_data or not self.or_reports_data:
            logger.error("Need to process both ultrasound and operative reports first")
            return None

        us_df = pd.DataFrame(self.us_reports_data)
        or_df = pd.DataFrame(self.or_reports_data)

        us_df['report_type'] = 'ultrasound'
        or_df['report_type'] = 'operative'

        # Merge on case_id instead of patient_id
        merged_df = pd.merge(
            us_df,
            or_df,
            on='case_id',
            how='inner',
            suffixes=('_us', '_or')
        )

        logger.info(f"Consolidated dataset created with {len(merged_df)} case records")
        return merged_df

    def run_full_extraction(self):
        config = self.config['data_extraction']

        # Download data from git repo
        self.download_data()

        # Process ultrasound reports
        logger.info("Starting ultrasound reports processing...")
        us_data = self.process_ultrasound_reports(config['us_pdf_directory'])
        if us_data:
            success = self.save_to_csv(us_data, config['us_output_csv'])
            if success:
                logger.info(f"Ultrasound reports saved to {config['us_output_csv']}")

        # Process operative reports
        logger.info("Starting operative reports processing...")
        or_data = self.process_operative_reports(config['or_pdf_directory'])
        if or_data:
            success = self.save_to_csv(or_data, config['or_output_csv'])
            if success:
                logger.info(f"Operative reports saved to {config['or_output_csv']}")

        # Generate consolidated dataset (merged_data.csv) based on case_id match
        if us_data and or_data:
            logger.info("Generating consolidated dataset by case_id...")
            consolidated_df = self.generate_consolidated_dataset()
            if consolidated_df is not None:
                success = self.save_to_csv(consolidated_df, config['consolidated_output_csv'])
                if success:
                    logger.info(f"Consolidated dataset saved to {config['consolidated_output_csv']}")
            else:
                logger.warning("Could not generate consolidated dataset - no matching case_id records")
        else:
            logger.warning("Insufficient data to generate consolidated dataset")

        logger.info("PDF data extraction completed!")


def main():
    extractor = PDFDataExtractor()
    extractor.run_full_extraction()


if __name__ == "__main__":
    main()