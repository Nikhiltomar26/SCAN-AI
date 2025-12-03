"""
model.py - ML Processing Module
Contains the MedicalReportProcessor class that handles OCR and local LLM analysis
"""

import os
from paddleocr import PaddleOCR
from PIL import Image
from groq import Groq
import traceback


class MedicalReportProcessor:
    """
    Encapsulates all ML logic for medical report processing:
    1. OCR extraction using PaddleOCR
    2. Text generation using local HuggingFace models (T5 or GPT-2)
    """
    
    def __init__(self, model_type: str = "t5"):
        """
        Initialize OCR and set up local LLM model
        
        Args:
            model_type: "t5" (Google T5) or "gpt2" (OpenAI GPT-2)
        """
        # Initialize PaddleOCR (English language, no angle classification for speed)
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

        # Use Groq API for LLM-based analysis
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client: {e}")

        print("Groq client initialized successfully")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using PaddleOCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            Exception: If OCR processing fails
        """
        try:
            # Verify image can be opened
            Image.open(image_path).verify()
            
            # Perform OCR
            result = self.ocr.ocr(image_path, cls=False)
            
            # Extract text from OCR result
            # PaddleOCR returns format: [[[box coordinates], (text, confidence)], ...]
            if not result or not result[0]:
                return ""
            
            extracted_lines = []
            for line in result[0]:
                text = line[1][0]  # Get text from tuple (text, confidence)
                extracted_lines.append(text)
            
            return "\n".join(extracted_lines)
        
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")

    def _get_groq_response(self, messages, temperature=0.7, max_tokens=1024):
        """Get response from Groq API with streaming."""
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=True,
                stop=None,
            )
            response_text = ""
            for chunk in completion:
                # defensive access to nested fields
                try:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        response_text += delta.content
                except Exception:
                    # some SDKs return dicts
                    try:
                        if chunk.get("choices") and chunk["choices"][0].get("delta") and chunk["choices"][0]["delta"].get("content"):
                            response_text += chunk["choices"][0]["delta"]["content"]
                    except Exception:
                        # fallback: try to append text if present
                        try:
                            if chunk.get("text"):
                                response_text += chunk.get("text")
                        except Exception:
                            pass

            return response_text.strip()
        except Exception as e:
            raise Exception(f"Groq API call failed: {str(e)}\n{traceback.format_exc()}")
    
    def analyze_with_t5(self, extracted_text: str) -> dict:
        """
        Use T5 model to generate explanation and highlights
        
        Args:
            extracted_text: Raw text extracted from the medical report
            
        Returns:
            Dictionary containing 'explanation' and 'highlights' keys
        """
        try:
            # Truncate if too long (T5-small has token limits)
            max_input_length = 512
            if len(extracted_text) > max_input_length:
                extracted_text = extracted_text[:max_input_length]
            
            # Task 1: Generate explanation
            explanation_prompt = f"explain in simple terms: {extracted_text}"
            explanation_inputs = self.tokenizer.encode(explanation_prompt, return_tensors="pt")
            explanation_outputs = self.model.generate(explanation_inputs, max_length=150, num_beams=4)
            explanation = self.tokenizer.decode(explanation_outputs[0], skip_special_tokens=True)
            
            # Task 2: Generate highlights
            highlights_prompt = f"summarize key points from: {extracted_text}"
            highlights_inputs = self.tokenizer.encode(highlights_prompt, return_tensors="pt")
            highlights_outputs = self.model.generate(highlights_inputs, max_length=100, num_beams=4)
            highlights_text = self.tokenizer.decode(highlights_outputs[0], skip_special_tokens=True)
            
            # Parse highlights into a list
            highlights = [h.strip() for h in highlights_text.split(",") if h.strip()]
            if not highlights:
                highlights = [highlights_text]
            
            # Limit to 5 highlights
            highlights = highlights[:5]
            
            return {
                "explanation": explanation if explanation else "Unable to generate explanation.",
                "highlights": highlights
            }
        
        except Exception as e:
            raise Exception(f"T5 analysis failed: {str(e)}")
    
    def analyze_with_gpt2(self, extracted_text: str) -> dict:
        """
        Use GPT-2 model to generate explanation and highlights
        
        Args:
            extracted_text: Raw text extracted from the medical report
            
        Returns:
            Dictionary containing 'explanation' and 'highlights' keys
        """
        try:
            # Truncate if too long (GPT-2 has token limits)
            max_input_length = 256
            if len(extracted_text) > max_input_length:
                extracted_text = extracted_text[:max_input_length]
            
            # Task 1: Generate explanation
            explanation_prompt = f"Medical Report Summary:\n{extracted_text}\n\nExplanation:"
            explanation_output = self.pipeline(explanation_prompt, do_sample=True, top_p=0.9)
            explanation = explanation_output[0]["generated_text"]
            
            # Extract only the generated part (remove prompt)
            if "Explanation:" in explanation:
                explanation = explanation.split("Explanation:")[-1].strip()
            
            # Task 2: Generate highlights
            highlights_prompt = f"Key findings from:\n{extracted_text}\n\nKey points:"
            highlights_output = self.pipeline(highlights_prompt, do_sample=True, top_p=0.9)
            highlights_text = highlights_output[0]["generated_text"]
            
            # Extract only the generated part
            if "Key points:" in highlights_text:
                highlights_text = highlights_text.split("Key points:")[-1].strip()
            
            # Parse highlights into a list
            highlights = [h.strip() for h in highlights_text.split("\n") if h.strip()]
            if not highlights:
                highlights = [highlights_text]
            
            # Limit to 5 highlights
            highlights = highlights[:5]
            
            return {
                "explanation": explanation if explanation else "Unable to generate explanation.",
                "highlights": highlights
            }
        
        except Exception as e:
            raise Exception(f"GPT-2 analysis failed: {str(e)}")
    
    def analyze_with_llm(self, extracted_text: str) -> dict:
        """
        Route to appropriate local model analysis
        
        Args:
            extracted_text: Raw text extracted from the medical report
            
        Returns:
            Dictionary containing 'explanation' and 'highlights' keys
        """
        # Use Groq chat-style API to get explanation and highlights
        try:
            system_msg = {"role": "system", "content": "You are a helpful medical assistant. Provide clear, non-alarming explanations for medical report contents and list key findings."}

            # Explanation
            explanation_prompt = (
                "Please read the following extracted medical report text and provide a plain-language explanation suitable for a patient. "
                "Be concise and avoid medical jargon where possible.\n\nReport Text:\n" + extracted_text
            )
            explanation_messages = [system_msg, {"role": "user", "content": explanation_prompt}]
            explanation = self._get_groq_response(explanation_messages, temperature=0.3, max_tokens=512)

            # Highlights
            highlights_prompt = (
                "Please list up to 5 key findings or points from the report. Return each item on a new line.\n\nReport Text:\n" + extracted_text
            )
            highlights_messages = [system_msg, {"role": "user", "content": highlights_prompt}]
            highlights_text = self._get_groq_response(highlights_messages, temperature=0.3, max_tokens=256)

            # Parse highlights into a list (split on newlines or commas)
            highlights = []
            if highlights_text:
                # prefer newlines
                if "\n" in highlights_text:
                    highlights = [h.strip('-• \t') for h in highlights_text.splitlines() if h.strip()]
                else:
                    highlights = [h.strip() for h in highlights_text.split(',') if h.strip()]

            if not highlights:
                highlights = [highlights_text] if highlights_text else []

            # Limit to 5
            highlights = highlights[:5]

            return {
                "explanation": explanation if explanation else "Unable to generate explanation.",
                "highlights": highlights
            }
        except Exception as e:
            raise Exception(f"LLM analysis (Groq) failed: {str(e)}")
    
    def process_medical_report(self, image_path: str) -> dict:
        """
        Complete pipeline: OCR extraction + local LLM analysis
        
        Args:
            image_path: Path to the medical report image
            
        Returns:
            Dictionary containing:
            - raw_text: Extracted text from OCR
            - explanation: Plain-language explanation from local model
            - highlights: Key highlights from local model
            
        Raises:
            Exception: If any step fails
        """
        # Step 1: Extract text
        raw_text = self.extract_text_from_image(image_path)
        
        if not raw_text.strip():
            return {
                "raw_text": "",
                "explanation": "No text could be extracted from the image.",
                "highlights": []
            }
        
        # Step 2: Analyze with local LLM
        analysis = self.analyze_with_llm(raw_text)
        
        return {
            "raw_text": raw_text,
            "explanation": analysis["explanation"],
            "highlights": analysis["highlights"]
        }