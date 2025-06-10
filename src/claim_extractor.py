import spacy
import re
from typing import List, Dict, Tuple
from transformers import pipeline
import numpy as np

class ClaimExtractor:
    def __init__(self):
        # Load spaCy model for NER and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize zero-shot classification pipeline for claim detection
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Enhanced candidate labels for better factual claim identification
        self.claim_labels = [
            "factual statement with evidence",
            "verifiable statistical claim", 
            "government or policy announcement",
            "official decision or action",
            "event or news report",
            "personal opinion or speculation",
            "hypothetical or future prediction",
            "unverifiable claim"
        ]
        
        # Factual claim indicators
        self.factual_patterns = [
            r'\b(announced|reported|confirmed|stated|declared|revealed|disclosed)\b',
            r'\b(government|ministry|department|agency|commission)\b',
            r'\b(policy|law|regulation|rule|mandate|requirement)\b',
            r'\b(launched|implemented|established|created|founded)\b',
            r'\b(study|research|survey|report|analysis)\b shows?\b',
            r'\b\d+(\.\d+)?%\b',  # Percentages
            r'\b\$\d+(\.\d+)?(k|m|million|billion|trillion)?\b',  # Money amounts
            r'\b\d+(\.\d+)?(k|m|million|billion|trillion)\b',  # Large numbers
        ]
        
        # Opinion/speculation indicators  
        self.opinion_patterns = [
            r'\b(think|believe|feel|opinion|suspect|guess|assume)\b',
            r'\b(should|could|would|might|may|probably|possibly|perhaps)\b',
            r'\b(in my view|personally|i believe|it seems)\b',
            r'\b(rumor|alleged|supposedly|reportedly without confirmation)\b'
        ]
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text with enhanced categorization"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        entity_confidence = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
                entity_confidence[ent.label_] = []
            entities[ent.label_].append(ent.text)
            # Use entity length and position as confidence proxy
            confidence = min(len(ent.text) / 20, 1.0) * (1.0 - abs(ent.start - len(doc) / 2) / len(doc))
            entity_confidence[ent.label_].append(confidence)
        
        return {"entities": entities, "confidence": entity_confidence}
    
    def extract_claims(self, text: str) -> List[Dict]:
        """Extract and classify potential factual claims with enhanced accuracy"""
        # Split text into sentences with better handling
        sentences = self._split_sentences_enhanced(text)
        
        claims = []
        for sentence in sentences:
            # Skip very short or uninformative sentences
            if len(sentence.strip()) < 15 or self._is_trivial_sentence(sentence):
                continue
                
            # Classify sentence to determine if it's a factual claim
            result = self.classifier(sentence, self.claim_labels)
            
            # Extract entities from the sentence
            entity_data = self.extract_entities(sentence)
            entities = entity_data.get("entities", {})
            
            # Enhanced claim scoring with multiple factors
            claim_score, score_details = self._enhanced_claim_scoring(result, entities, sentence)
            
            # Determine verifiability with more sophisticated logic
            is_verifiable = self._assess_verifiability(result, entities, sentence, claim_score)
            
            claim_data = {
                "text": sentence.strip(),
                "classification": result["labels"][0],
                "classification_confidence": result["scores"][0],
                "all_classifications": dict(zip(result["labels"], result["scores"])),
                "entities": entities,
                "claim_score": claim_score,
                "score_breakdown": score_details,
                "is_verifiable": is_verifiable,
                "verifiability_confidence": self._calculate_verifiability_confidence(claim_score, entities, sentence)
            }
            
            claims.append(claim_data)
        
        # Enhanced sorting considering multiple factors
        claims.sort(key=lambda x: (x["is_verifiable"], x["claim_score"], x["classification_confidence"]), reverse=True)
        
        # Apply claim deduplication and filtering
        claims = self._deduplicate_claims(claims)
        
        return claims
    
    def _split_sentences_enhanced(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better handling of complex structures"""
        if self.nlp:
            doc = self.nlp(text)
            sentences = []
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                # Split compound sentences with conjunctions
                if len(sentence_text) > 100 and any(conj in sentence_text.lower() for conj in [' and ', ' but ', ' however ', ' moreover ', ' furthermore ']):
                    sub_sentences = self._split_compound_sentence(sentence_text)
                    sentences.extend(sub_sentences)
                else:
                    sentences.append(sentence_text)
            return sentences
        else:
            # Enhanced fallback regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _split_compound_sentence(self, sentence: str) -> List[str]:
        """Split compound sentences while preserving meaning"""
        # Simple conjunction-based splitting
        conjunctions = [' and ', ' but ', ' however ', ' moreover ', ' furthermore ']
        for conj in conjunctions:
            if conj in sentence.lower():
                parts = sentence.split(conj, 1)
                if len(parts) == 2 and len(parts[0].strip()) > 20 and len(parts[1].strip()) > 20:
                    return [parts[0].strip(), parts[1].strip()]
        return [sentence]
    
    def _is_trivial_sentence(self, sentence: str) -> bool:
        """Check if sentence is too trivial or generic to fact-check"""
        trivial_patterns = [
            r'^(yes|no|ok|okay|sure|thanks|hello|hi)\b',
            r'^\w+\s*[.!?]*$',  # Single word sentences
            r'^(this|that|it|he|she|they)\s+(is|was|are|were)\s+\w+\s*[.!?]*$',  # Vague pronouns
        ]
        
        sentence_lower = sentence.lower().strip()
        return any(re.match(pattern, sentence_lower) for pattern in trivial_patterns)
    
    def _enhanced_claim_scoring(self, classification_result: Dict, entities: Dict, text: str) -> Tuple[float, Dict]:
        """Enhanced claim scoring with detailed breakdown"""
        score_details = {}
        
        # 1. Base classification score (weighted by factual vs opinion labels)
        top_label = classification_result["labels"][0]
        top_confidence = classification_result["scores"][0]
        
        if "opinion" in top_label.lower() or "speculation" in top_label.lower() or "prediction" in top_label.lower():
            classification_score = top_confidence * 0.2  # Lower weight for opinions
        elif "unverifiable" in top_label.lower():
            classification_score = top_confidence * 0.1
        else:
            classification_score = top_confidence * 0.6  # Higher weight for factual claims
        
        score_details["classification"] = classification_score
        
        # 2. Entity-based scoring with more sophisticated weighting
        entity_score = self._calculate_entity_score(entities)
        score_details["entities"] = entity_score
        
        # 3. Pattern-based scoring
        factual_pattern_score = self._calculate_pattern_score(text, self.factual_patterns, positive=True)
        opinion_pattern_score = self._calculate_pattern_score(text, self.opinion_patterns, positive=False)
        pattern_score = factual_pattern_score - opinion_pattern_score
        score_details["patterns"] = pattern_score
        
        # 4. Linguistic complexity and specificity
        specificity_score = self._calculate_specificity_score(text, entities)
        score_details["specificity"] = specificity_score
        
        # 5. Numerical content scoring
        numerical_score = self._calculate_numerical_score(text)
        score_details["numerical"] = numerical_score
        
        # 6. Temporal indicators
        temporal_score = self._calculate_temporal_score(text)
        score_details["temporal"] = temporal_score
        
        # Combine scores with weights
        total_score = (
            classification_score * 0.35 +
            entity_score * 0.25 +
            pattern_score * 0.15 +
            specificity_score * 0.1 +
            numerical_score * 0.1 +
            temporal_score * 0.05
        )
        
        return min(max(total_score, 0.0), 1.0), score_details
    
    def _calculate_entity_score(self, entities: Dict) -> float:
        """Calculate score based on entity types and their factual relevance"""
        if not entities:
            return 0.0
        
        # Weight different entity types by their factual importance
        entity_weights = {
            "ORG": 0.15,      # Organizations
            "GPE": 0.12,      # Countries, cities
            "PERSON": 0.1,    # People
            "DATE": 0.15,     # Dates
            "MONEY": 0.18,    # Money amounts
            "PERCENT": 0.18,  # Percentages
            "CARDINAL": 0.12, # Numbers
            "EVENT": 0.15,    # Events
            "LAW": 0.17,      # Laws, regulations
            "PRODUCT": 0.08,  # Products
            "WORK_OF_ART": 0.05,
            "LANGUAGE": 0.05,
            "TIME": 0.1,
            "QUANTITY": 0.12
        }
        
        score = 0.0
        for entity_type, entity_list in entities.items():
            weight = entity_weights.get(entity_type, 0.05)
            # Score based on number of entities, with diminishing returns
            count_score = min(len(entity_list) * 0.3, 1.0)
            score += weight * count_score
        
        return min(score, 0.5)  # Cap entity contribution
    
    def _calculate_pattern_score(self, text: str, patterns: List[str], positive: bool = True) -> float:
        """Calculate score based on pattern matching"""
        text_lower = text.lower()
        matches = 0
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches += 1
        
        # Normalize by text length and pattern count
        score = min(matches * 0.1, 0.3)
        return score if positive else -score
    
    def _calculate_specificity_score(self, text: str, entities: Dict) -> float:
        """Calculate score based on text specificity and detail level"""
        # Factors: proper nouns, specific terms, detail level
        word_count = len(text.split())
        if word_count < 5:
            return 0.0
        
        # Count proper nouns and specific terms
        proper_nouns = sum(len(ents) for ents in entities.values())
        specificity_indicators = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        
        # Normalize scores
        proper_noun_score = min(proper_nouns / word_count, 0.3)
        specificity_score = min(specificity_indicators / word_count, 0.2)
        
        return proper_noun_score + specificity_score
    
    def _calculate_numerical_score(self, text: str) -> float:
        """Score based on presence and context of numbers"""
        # Look for numbers in factual contexts
        numerical_patterns = [
            r'\b\d+(\.\d+)?%\b',  # Percentages
            r'\b\$\d+(\.\d+)?([km]|million|billion)?\b',  # Money
            r'\b\d{4}\b',  # Years
            r'\b\d+(\.\d+)?\s*(million|billion|trillion|thousand)\b',  # Large numbers
            r'\b\d+(\.\d+)?\s*(km|miles|meters|feet|pounds|kg|tons)\b',  # Measurements
        ]
        
        score = 0.0
        for pattern in numerical_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.1
        
        return min(score, 0.3)
    
    def _calculate_temporal_score(self, text: str) -> float:
        """Score based on temporal specificity"""
        temporal_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(today|yesterday|tomorrow|this week|last week|next week)\b',
            r'\b(2019|2020|2021|2022|2023|2024|2025)\b'
        ]
        
        score = 0.0
        for pattern in temporal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.05
        
        return min(score, 0.2)
    
    def _assess_verifiability(self, classification_result: Dict, entities: Dict, text: str, claim_score: float) -> bool:
        """Enhanced verifiability assessment"""
        # Multiple criteria for verifiability
        criteria_met = 0
        
        # 1. Classification suggests factual content
        top_label = classification_result["labels"][0].lower()
        if not any(term in top_label for term in ["opinion", "speculation", "unverifiable", "prediction"]):
            criteria_met += 1
        
        # 2. Has important entities
        if any(entity_type in entities for entity_type in ["ORG", "GPE", "DATE", "MONEY", "PERCENT", "EVENT", "LAW"]):
            criteria_met += 1
        
        # 3. Contains factual patterns
        if any(re.search(pattern, text.lower()) for pattern in self.factual_patterns):
            criteria_met += 1
        
        # 4. Doesn't contain too many opinion patterns
        opinion_matches = sum(1 for pattern in self.opinion_patterns if re.search(pattern, text.lower()))
        if opinion_matches < 2:
            criteria_met += 1
        
        # 5. Has sufficient claim score
        if claim_score > 0.4:
            criteria_met += 1
        
        # 6. Has specific, detailed content
        if len(text.split()) > 8 and re.search(r'[A-Z][a-z]+', text):
            criteria_met += 1
        
        # Require at least 3 out of 6 criteria
        return criteria_met >= 3
    
    def _calculate_verifiability_confidence(self, claim_score: float, entities: Dict, text: str) -> float:
        """Calculate confidence in the verifiability assessment"""
        confidence_factors = [
            min(claim_score * 2, 1.0),  # Claim score contribution
            min(len(entities) * 0.2, 0.4),  # Entity diversity contribution
            min(len(text.split()) / 20, 0.3),  # Text length contribution
            0.3 if re.search(r'\d+', text) else 0.0  # Numerical content
        ]
        
        return min(sum(confidence_factors), 1.0)
    
    def _deduplicate_claims(self, claims: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar claims"""
        if len(claims) <= 1:
            return claims
        
        deduplicated = []
        
        for i, claim in enumerate(claims):
            is_duplicate = False
            claim_text = claim["text"].lower().strip()
            
            for existing_claim in deduplicated:
                existing_text = existing_claim["text"].lower().strip()
                
                # Check for exact duplicates
                if claim_text == existing_text:
                    is_duplicate = True
                    break
                
                # Check for high similarity (simple word overlap)
                claim_words = set(claim_text.split())
                existing_words = set(existing_text.split())
                
                if len(claim_words) > 3 and len(existing_words) > 3:
                    overlap = len(claim_words.intersection(existing_words))
                    similarity = overlap / min(len(claim_words), len(existing_words))
                    
                    if similarity > 0.8:  # 80% word overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(claim)
        
        return deduplicated
    
    def get_primary_claim(self, text: str) -> Dict:
        """Extract the primary verifiable claim with enhanced selection logic"""
        claims = self.extract_claims(text)
        
        # Filter verifiable claims first
        verifiable_claims = [c for c in claims if c["is_verifiable"]]
        
        if verifiable_claims:
            # Select based on multiple factors
            best_claim = max(verifiable_claims, key=lambda c: (
                c["claim_score"] * 0.4 +
                c["classification_confidence"] * 0.3 +
                c["verifiability_confidence"] * 0.3
            ))
            return best_claim
        elif claims:
            # Fallback to best overall claim
            return max(claims, key=lambda c: c["claim_score"])
        else:
            return {
                "text": text,
                "classification": "unverifiable",
                "classification_confidence": 0.0,
                "entities": {},
                "claim_score": 0.0,
                "score_breakdown": {},
                "is_verifiable": False,
                "verifiability_confidence": 0.0
            } 