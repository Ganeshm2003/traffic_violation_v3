import cv2
import numpy as np
import re
from models.detection_models import ModelManager
from config.settings import Config

class LicensePlateDetector:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def detect_license_plate(self, frame, vehicle_bbox):
        """Extract license plate text from vehicle with enhanced preprocessing"""
        try:
            ocr_reader = self.model_manager.get_ocr_reader()
            x1, y1, x2, y2 = vehicle_bbox
            
            # Extract vehicle ROI with padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                return ""
            
            # Focus on the lower part of vehicle where license plates are typically located
            height = vehicle_roi.shape[0]
            lower_roi = vehicle_roi[int(height*0.6):, :]
            
            # Try multiple preprocessing approaches
            license_candidates = []
            
            # Method 1: Original image
            candidates = self._extract_text_with_ocr(ocr_reader, lower_roi)
            license_candidates.extend(candidates)
            
            # Method 2: Enhanced preprocessing
            try:
                enhanced_roi = self._preprocess_for_ocr(lower_roi)
                candidates = self._extract_text_with_ocr(ocr_reader, enhanced_roi)
                license_candidates.extend(candidates)
            except Exception as e:
                print(f"Error in enhanced preprocessing: {e}")
            
            # Return the best candidate
            return self._select_best_license_plate(license_candidates)
            
        except Exception as e:
            print(f"License plate detection error: {e}")
            return ""
    
    def _preprocess_for_ocr(self, roi):
        """Enhanced preprocessing for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize for better OCR (if image is too small)
            height, width = processed.shape
            if height < 50 or width < 150:
                scale_factor = max(2, 150 // width) if width > 0 else 2
                processed = cv2.resize(processed, (width * scale_factor, height * scale_factor),
                                     interpolation=cv2.INTER_CUBIC)
            
            return processed
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return roi
    
    def _extract_text_with_ocr(self, ocr_reader, image):
        """Extract text using OCR with improved settings"""
        try:
            results = ocr_reader.readtext(
                image,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                width_ths=0.7,
                height_ths=0.7,
                paragraph=False
            )
            
            candidates = []
            for (bbox, text, conf) in results:
                if conf > Config.LICENSE_PLATE_CONFIDENCE_THRESHOLD:
                    clean_text = self._clean_license_text(text)
                    if self._is_valid_indian_license_plate(clean_text):
                        candidates.append((clean_text, conf))
            
            return candidates
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return []
    
    def _clean_license_text(self, text):
        """Clean and format license plate text"""
        try:
            # Remove non-alphanumeric characters
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Common OCR corrections for Indian license plates
            corrections = {
                'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8'
            }
            
            # Apply corrections cautiously
            result = ""
            for char in clean_text:
                if char.isdigit():
                    result += char
                elif char.isalpha():
                    result += corrections.get(char, char)
                else:
                    result += char
            
            return result
        except Exception as e:
            print(f"Error cleaning license text: {e}")
            return text.upper()
    
    def _is_valid_indian_license_plate(self, text):
        """Validate if text matches Indian license plate patterns"""
        if not text or len(text) < 6:
            return False
        
        try:
            # Indian license plate patterns (simplified)
            patterns = [
                r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$',  # Standard format
                r'^[A-Z]{1,3}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$',  # Variations
            ]
            
            for pattern in patterns:
                if re.match(pattern, text):
                    return True
            
            # Fallback: reasonable length with mix of letters and numbers
            if 6 <= len(text) <= 12:
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                return has_letters and has_numbers
            
            return False
        except Exception as e:
            print(f"Error validating license plate: {e}")
            return len(text) >= 6
    
    def _select_best_license_plate(self, candidates):
        """Select the best license plate candidate from all methods"""
        if not candidates:
            return ""
        
        try:
            # Remove duplicates and sort by confidence
            unique_candidates = {}
            for text, conf in candidates:
                if text not in unique_candidates or conf > unique_candidates[text]:
                    unique_candidates[text] = conf
            
            if not unique_candidates:
                return ""
            
            # Return the candidate with highest confidence
            best_candidate = max(unique_candidates.items(), key=lambda x: x[1])
            return best_candidate[0] if best_candidate[1] > Config.LICENSE_PLATE_CONFIDENCE_THRESHOLD else ""
            
        except Exception as e:
            print(f"Error selecting best candidate: {e}")
            return candidates[0][0] if candidates else ""