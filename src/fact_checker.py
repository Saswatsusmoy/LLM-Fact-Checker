import json
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import re

class FactChecker:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the fact checker with a local LLM
        
        Args:
            model_name: Name of the Hugging Face model to use for fact checking
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the text generation pipeline with a smaller, faster model
        print(f"Loading fact-checking model: {model_name}")
        try:
            # Use a smaller, more suitable model for text generation
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple rule-based checking...")
            self.generator = None
        
        # Verdict mappings
        self.verdicts = {
            "TRUE": "✅ True",
            "FALSE": "❌ False", 
            "UNVERIFIABLE": "🤷‍♂️ Unverifiable"
        }
    
    def check_claim(self, claim: str, evidence_facts: List[Dict], similarity_threshold: float = 0.2) -> Dict:
        """
        Check a claim against evidence facts and return a verdict
        
        Args:
            claim: The claim to fact-check
            evidence_facts: List of relevant facts from the database
            similarity_threshold: Minimum similarity score for evidence to be considered
            
        Returns:
            Dictionary with verdict, evidence, reasoning, and confidence
        """
        if not evidence_facts:
            return self._create_unverifiable_result(claim, "No relevant evidence found in database")
        
        # Filter facts by similarity score
        relevant_facts = [
            fact for fact in evidence_facts 
            if fact.get('similarity_score', 0) >= similarity_threshold
        ]
        
        if not relevant_facts:
            return self._create_unverifiable_result(claim, f"No evidence above similarity threshold {similarity_threshold}")
        
        # Use LLM for comparison if available, otherwise use rule-based approach
        if self.generator:
            return self._llm_fact_check(claim, relevant_facts)
        else:
            return self._rule_based_fact_check(claim, relevant_facts)
    
    def _llm_fact_check(self, claim: str, evidence_facts: List[Dict]) -> Dict:
        """Enhanced LLM-based fact checking with sophisticated reasoning"""
        try:
            # Determine claim type for specialized prompting
            claim_type = self._analyze_claim_type(claim)
            
            # Enhanced evidence preparation with similarity analysis
            evidence_analysis = self._analyze_evidence_quality(evidence_facts)
            formatted_evidence = self._format_evidence_with_analysis(evidence_facts, evidence_analysis)
            
            # Create sophisticated prompt based on claim type and evidence quality
            prompt = self._create_enhanced_prompt(claim, claim_type, formatted_evidence, evidence_analysis)
            
            # Try detailed analysis first
            try:
                response = self.generator(
                    prompt,
                    max_new_tokens=300,  # Increased for better reasoning
                    num_return_sequences=1,
                    pad_token_id=50256,
                    do_sample=True,
                    temperature=0.3,  # Lower temperature for more focused reasoning
                    top_p=0.9,
                    truncation=True
                )
                
                generated_text = response[0]['generated_text']
                response_text = generated_text[len(prompt):].strip()
                
                # Enhanced response parsing
                verdict_info = self._parse_enhanced_llm_response(response_text, claim, evidence_facts, evidence_analysis)
                verdict_info['claim_type'] = claim_type
                verdict_info['evidence_analysis'] = evidence_analysis
                
                return verdict_info
                
            except Exception as token_error:
                print(f"Detailed LLM analysis failed, trying concise approach: {token_error}")
                return self._fallback_llm_analysis(claim, evidence_facts, claim_type)
            
        except Exception as e:
            print(f"Error in enhanced LLM fact-checking: {e}")
            return self._rule_based_fact_check(claim, evidence_facts)
    
    def _analyze_claim_type(self, claim: str) -> str:
        """Analyze the type of claim for specialized handling"""
        claim_lower = claim.lower()
        
        # Define claim type patterns
        claim_patterns = {
            'statistical': [r'\d+%', r'\d+\.\d+%', r'\$\d+', r'\d+ million', r'\d+ billion', r'increased by', r'decreased by'],
            'temporal': [r'in \d{4}', r'since \d{4}', r'by \d{4}', r'january|february|march|april|may|june|july|august|september|october|november|december'],
            'factual_entity': [r'announced', r'confirmed', r'reported', r'stated', r'declared'],
            'policy': [r'policy', r'law', r'regulation', r'bill', r'act', r'legislation'],
            'scientific': [r'study', r'research', r'scientist', r'university', r'published', r'journal'],
            'prediction': [r'will', r'going to', r'expected to', r'planned', r'scheduled'],
            'opinion': [r'believe', r'think', r'opinion', r'should', r'better', r'worse', r'good', r'bad']
        }
        
        for claim_type, patterns in claim_patterns.items():
            if any(re.search(pattern, claim_lower) for pattern in patterns):
                return claim_type
        
        return 'general'
    
    def _analyze_evidence_quality(self, evidence_facts: List[Dict]) -> Dict:
        """Analyze the quality and characteristics of evidence"""
        if not evidence_facts:
            return {'quality_score': 0, 'diversity_score': 0, 'reliability_indicators': []}
        
        analysis = {
            'total_evidence': len(evidence_facts),
            'avg_similarity': sum(fact.get('similarity_score', 0) for fact in evidence_facts) / len(evidence_facts),
            'max_similarity': max(fact.get('similarity_score', 0) for fact in evidence_facts),
            'min_similarity': min(fact.get('similarity_score', 0) for fact in evidence_facts),
            'unique_sources': len(set(fact.get('source', 'Unknown') for fact in evidence_facts)),
            'content_lengths': [len(fact.get('content', '')) for fact in evidence_facts],
            'recency_scores': []
        }
        
        # Calculate quality indicators
        analysis['avg_content_length'] = sum(analysis['content_lengths']) / len(analysis['content_lengths'])
        analysis['source_diversity'] = analysis['unique_sources'] / len(evidence_facts)
        
        # Quality score based on multiple factors
        quality_factors = [
            min(analysis['avg_similarity'] * 2, 1.0),  # Similarity factor
            min(analysis['source_diversity'] * 2, 1.0),  # Source diversity
            min(analysis['avg_content_length'] / 200, 1.0),  # Content richness
            min(len(evidence_facts) / 5, 1.0)  # Evidence quantity
        ]
        
        analysis['quality_score'] = sum(quality_factors) / len(quality_factors)
        analysis['diversity_score'] = analysis['source_diversity']
        
        # Identify reliability indicators
        reliability_indicators = []
        if analysis['max_similarity'] > 0.8:
            reliability_indicators.append('high_similarity_match')
        if analysis['unique_sources'] >= 3:
            reliability_indicators.append('diverse_sources')
        if analysis['avg_content_length'] > 150:
            reliability_indicators.append('detailed_content')
        
        analysis['reliability_indicators'] = reliability_indicators
        
        return analysis
    
    def _format_evidence_with_analysis(self, evidence_facts: List[Dict], evidence_analysis: Dict) -> str:
        """Format evidence with enhanced analysis for better LLM reasoning"""
        if not evidence_facts:
            return "No relevant evidence found in database."
        
        formatted_evidence = []
        formatted_evidence.append(f"EVIDENCE SUMMARY: {len(evidence_facts)} sources, avg similarity: {evidence_analysis['avg_similarity']:.3f}")
        formatted_evidence.append(f"SOURCE DIVERSITY: {evidence_analysis['unique_sources']} unique sources")
        formatted_evidence.append("")
        
        for i, fact in enumerate(evidence_facts[:5], 1):  # Limit to top 5 for token efficiency
            source = fact.get('source', 'Unknown')
            similarity = fact.get('similarity_score', 0)
            content = fact.get('content', '')[:400]  # Limit content length
            
            formatted_evidence.append(f"EVIDENCE {i}:")
            formatted_evidence.append(f"  Source: {source}")
            formatted_evidence.append(f"  Similarity: {similarity:.3f}")
            formatted_evidence.append(f"  Content: {content}")
            formatted_evidence.append("")
        
        return '\n'.join(formatted_evidence)
    
    def _create_enhanced_prompt(self, claim: str, claim_type: str, formatted_evidence: str, evidence_analysis: Dict) -> str:
        """Create sophisticated prompt based on claim type and evidence quality"""
        
        # Base instructions with claim-type specific guidance
        type_specific_instructions = {
            'statistical': "Pay special attention to numerical accuracy, data sources, and statistical methodology.",
            'temporal': "Focus on dates, timelines, and temporal accuracy. Check if events occurred as stated.",
            'factual_entity': "Verify entity names, their roles, and the accuracy of statements attributed to them.",
            'policy': "Examine policy details, implementation status, and official sources.",
            'scientific': "Evaluate research methodology, peer review status, and scientific consensus.",
            'prediction': "Assess the basis for predictions and distinguish between plans and speculation.",
            'opinion': "Recognize this as opinion-based and evaluate if it's presented as factual.",
            'general': "Conduct comprehensive fact-checking across all relevant dimensions."
        }
        
        instruction = type_specific_instructions.get(claim_type, type_specific_instructions['general'])
        
        # Quality-based confidence guidance
        quality_guidance = ""
        if evidence_analysis['quality_score'] > 0.7:
            quality_guidance = "HIGH QUALITY EVIDENCE: Multiple reliable sources with strong similarity. High confidence verdict appropriate."
        elif evidence_analysis['quality_score'] > 0.4:
            quality_guidance = "MODERATE QUALITY EVIDENCE: Some reliable sources. Medium confidence verdict recommended."
        else:
            quality_guidance = "LIMITED QUALITY EVIDENCE: Few or weak sources. Lower confidence verdict suggested."
        
        prompt = f"""You are an expert fact-checker analyzing the following claim with available evidence.

CLAIM TO VERIFY: {claim}
CLAIM TYPE: {claim_type.upper()}

SPECIAL INSTRUCTIONS: {instruction}
EVIDENCE QUALITY: {quality_guidance}

{formatted_evidence}

ANALYSIS FRAMEWORK:
1. CONTENT ANALYSIS: How well does the evidence address the specific claim?
2. SOURCE RELIABILITY: Evaluate the credibility and diversity of sources
3. CONSISTENCY CHECK: Do multiple sources agree or contradict?
4. SPECIFICITY MATCH: Does the evidence match the specific details in the claim?
5. TEMPORAL RELEVANCE: Is the evidence current and relevant to the timeframe?

Please provide your analysis in this EXACT format:

ANALYSIS: [Provide detailed reasoning about evidence quality, source reliability, and content match]

VERDICT: [Choose exactly one: TRUE, FALSE, or UNVERIFIABLE]

REASONING: [Explain your verdict based on the evidence analysis above]

CONFIDENCE: [Choose exactly one: HIGH, MEDIUM, or LOW]"""

        return prompt
    
    def _fallback_llm_analysis(self, claim: str, evidence_facts: List[Dict], claim_type: str) -> Dict:
        """Fallback LLM analysis with minimal token usage"""
        try:
            # Create minimal but effective prompt
            evidence_summary = self._create_evidence_summary(evidence_facts)
            
            minimal_prompt = f"""Fact-check this claim with evidence:
CLAIM: {claim[:150]}
EVIDENCE: {evidence_summary[:200]}
VERDICT (TRUE/FALSE/UNVERIFIABLE): """

            response = self.generator(
                minimal_prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=50256,
                do_sample=False,  # Greedy for consistency
                truncation=True
            )
            
            generated_text = response[0]['generated_text']
            response_text = generated_text[len(minimal_prompt):].strip()
            
            # Simple verdict extraction
            verdict = "UNVERIFIABLE"
            confidence_level = "LOW"
            
            if "TRUE" in response_text.upper() and "FALSE" not in response_text.upper():
                verdict = "TRUE"
                confidence_level = "MEDIUM"
            elif "FALSE" in response_text.upper():
                verdict = "FALSE"
                confidence_level = "MEDIUM"
            
            confidence = 0.6 if confidence_level == "MEDIUM" else 0.3
            
            return {
                "verdict": self.verdicts[verdict],
                "raw_verdict": verdict,
                "confidence": confidence,
                "evidence": [fact.get('content', '') for fact in evidence_facts[:2]],
                "evidence_sources": [fact.get('source', 'Unknown') for fact in evidence_facts[:2]],
                "reasoning": f"Fallback analysis: {response_text[:200]}",
                "similarity_scores": [fact.get('similarity_score', 0) for fact in evidence_facts[:2]],
                "confidence_level": confidence_level,
                "analysis": "Token-constrained fallback analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Fallback LLM analysis failed: {e}")
            return self._rule_based_fact_check(claim, evidence_facts)
    
    def _create_evidence_summary(self, evidence_facts: List[Dict]) -> str:
        """Create concise evidence summary for token-efficient prompts"""
        if not evidence_facts:
            return "No evidence"
        
        # Get top evidence by similarity
        top_evidence = sorted(evidence_facts, key=lambda x: x.get('similarity_score', 0), reverse=True)[:2]
        
        summaries = []
        for fact in top_evidence:
            source = fact.get('source', 'Unknown')
            content = fact.get('content', '')[:100]
            similarity = fact.get('similarity_score', 0)
            summaries.append(f"{source} ({similarity:.2f}): {content}")
        
        return ' | '.join(summaries)
    
    def _parse_enhanced_llm_response(self, response: str, claim: str, evidence_facts: List[Dict], evidence_analysis: Dict) -> Dict:
        """Parse enhanced LLM response with sophisticated reasoning analysis"""
        response_lines = response.strip().split('\n')
        
        # Initialize defaults
        verdict = "UNVERIFIABLE"
        reasoning = "Unable to parse LLM response properly."
        analysis = ""
        confidence_level = "LOW"
        
        # Parse structured response
        current_section = None
        analysis_lines = []
        reasoning_lines = []
        
        for line in response_lines:
            line = line.strip()
            if line.startswith("ANALYSIS:"):
                current_section = "analysis"
                analysis_lines.append(line[9:].strip())
            elif line.startswith("VERDICT:"):
                current_section = "verdict"
                verdict_text = line[8:].strip().upper()
                if "TRUE" in verdict_text and "FALSE" not in verdict_text:
                    verdict = "TRUE"
                elif "FALSE" in verdict_text:
                    verdict = "FALSE"
                else:
                    verdict = "UNVERIFIABLE"
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning_lines.append(line[10:].strip())
            elif line.startswith("CONFIDENCE:"):
                confidence_text = line[11:].strip().upper()
                if "HIGH" in confidence_text:
                    confidence_level = "HIGH"
                elif "MEDIUM" in confidence_text:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
            elif current_section == "analysis" and line:
                analysis_lines.append(line)
            elif current_section == "reasoning" and line:
                reasoning_lines.append(line)
        
        # Combine parsed sections
        analysis = ' '.join(analysis_lines).strip()
        reasoning = ' '.join(reasoning_lines).strip()
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(
            verdict, confidence_level, evidence_analysis, analysis, reasoning
        )
        
        # Fallback parsing if structured format wasn't followed
        if not reasoning and not analysis:
            reasoning = response.strip()
            analysis = "LLM provided unstructured response"
        
        # Clean up reasoning and analysis
        reasoning = re.sub(r'\s+', ' ', reasoning).strip()
        analysis = re.sub(r'\s+', ' ', analysis).strip()
        
        if len(reasoning) > 400:
            reasoning = reasoning[:400] + "..."
        
        # Combine analysis and reasoning for comprehensive explanation
        if analysis:
            full_reasoning = f"Analysis: {analysis[:150]}... Conclusion: {reasoning}"
            if len(full_reasoning) <= 500:
                reasoning = full_reasoning
        
        return {
            "verdict": self.verdicts[verdict],
            "raw_verdict": verdict,
            "confidence": confidence,
            "evidence": [fact.get('content', '') for fact in evidence_facts[:3]],
            "evidence_sources": [fact.get('source', 'Unknown') for fact in evidence_facts[:3]],
            "reasoning": reasoning,
            "similarity_scores": [fact.get('similarity_score', 0) for fact in evidence_facts],
            "confidence_level": confidence_level,
            "analysis": analysis,
            "evidence_quality": evidence_analysis['quality_score'],
            "source_diversity": evidence_analysis['diversity_score'],
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_enhanced_confidence(self, verdict: str, confidence_level: str, evidence_analysis: Dict, analysis: str, reasoning: str) -> float:
        """Calculate confidence with enhanced factors"""
        # Base confidence from LLM
        base_confidence = {
            "HIGH": 0.85,
            "MEDIUM": 0.65,
            "LOW": 0.35
        }.get(confidence_level, 0.5)
        
        # Evidence quality adjustment
        quality_adjustment = (evidence_analysis['quality_score'] - 0.5) * 0.2
        
        # Source diversity adjustment
        diversity_adjustment = evidence_analysis['diversity_score'] * 0.1
        
        # Reasoning quality adjustment based on text analysis
        reasoning_quality = self._assess_reasoning_quality(analysis, reasoning)
        reasoning_adjustment = (reasoning_quality - 0.5) * 0.15
        
        # Similarity-based adjustment
        avg_similarity = evidence_analysis['avg_similarity']
        similarity_adjustment = (avg_similarity - 0.5) * 0.1
        
        # Calculate final confidence
        final_confidence = base_confidence + quality_adjustment + diversity_adjustment + reasoning_adjustment + similarity_adjustment
        
        # Ensure confidence is within bounds
        return max(min(final_confidence, 0.98), 0.02)
    
    def _assess_reasoning_quality(self, analysis: str, reasoning: str) -> float:
        """Assess the quality of LLM reasoning"""
        combined_text = f"{analysis} {reasoning}".lower()
        
        quality_indicators = [
            'evidence', 'source', 'reliable', 'consistent', 'contradictory',
            'specific', 'detailed', 'multiple', 'confirmed', 'verified'
        ]
        
        quality_score = 0.5  # Base score
        
        # Count quality indicators
        indicator_count = sum(1 for indicator in quality_indicators if indicator in combined_text)
        quality_score += min(indicator_count * 0.05, 0.3)
        
        # Length factor (more detailed reasoning generally better)
        text_length = len(combined_text.split())
        if text_length > 20:
            quality_score += 0.1
        elif text_length < 10:
            quality_score -= 0.1
        
        # Specificity indicators
        if any(phrase in combined_text for phrase in ['specific evidence', 'multiple sources', 'cross-verified']):
            quality_score += 0.1
        
        return max(min(quality_score, 1.0), 0.0)
    
    def _rule_based_fact_check(self, claim: str, evidence_facts: List[Dict]) -> Dict:
        """Enhanced rule-based fact checking as fallback"""
        claim_lower = claim.lower()
        
        # Analyze similarity scores and content overlap
        avg_similarity = sum(fact.get('similarity_score', 0) for fact in evidence_facts) / len(evidence_facts)
        max_similarity = max(fact.get('similarity_score', 0) for fact in evidence_facts)
        
        # Enhanced keyword analysis with more nuanced categorization
        contradiction_keywords = [
            'not', 'false', 'denied', 'rejected', 'cancelled', 'postponed', 'debunked', 
            'refuted', 'incorrect', 'wrong', 'untrue', 'hoax', 'fake', 'misinformation'
        ]
        confirmation_keywords = [
            'confirmed', 'announced', 'approved', 'launched', 'implemented', 'started',
            'verified', 'validated', 'true', 'accurate', 'factual', 'established', 'proven'
        ]
        uncertainty_keywords = [
            'alleged', 'rumored', 'unconfirmed', 'speculated', 'claimed', 'reportedly',
            'possibly', 'potentially', 'may have', 'might be', 'uncertain'
        ]
        
        evidence_contents = [fact.get('content', '').lower() for fact in evidence_facts]
        
        # More sophisticated keyword scoring
        contradiction_score = 0
        confirmation_score = 0
        uncertainty_score = 0
        
        for content in evidence_contents:
            for keyword in contradiction_keywords:
                if keyword in content:
                    # Give more weight to stronger contradiction words
                    weight = 2 if keyword in ['false', 'debunked', 'hoax'] else 1
                    contradiction_score += weight
            
            for keyword in confirmation_keywords:
                if keyword in content:
                    # Give more weight to stronger confirmation words
                    weight = 2 if keyword in ['confirmed', 'verified', 'proven'] else 1
                    confirmation_score += weight
            
            for keyword in uncertainty_keywords:
                if keyword in content:
                    uncertainty_score += 1
        
        # Analyze source diversity and recency
        unique_sources = len(set(fact.get('source', '') for fact in evidence_facts))
        source_diversity_bonus = min(unique_sources * 0.05, 0.2)  # Max 20% bonus
        
        # Check for numerical/factual claims that require precise matching
        claim_has_numbers = bool(re.search(r'\d+', claim))
        claim_has_dates = bool(re.search(r'\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', claim_lower))
        
        # Calculate base confidence
        confidence = avg_similarity + source_diversity_bonus
        
        # Determine verdict with enhanced logic
        reasoning_parts = []
        
        if max_similarity > 0.8 and confirmation_score > contradiction_score * 1.2:  # Lowered threshold from 1.5
            verdict = "TRUE"
            confidence_level = "HIGH" if confidence > 0.7 else "MEDIUM"  # Lowered from 0.8
            reasoning_parts.append(f"Very high similarity match (max: {max_similarity:.3f})")
            reasoning_parts.append(f"Strong confirmation signals ({confirmation_score} vs {contradiction_score} contradictions)")
            
        elif avg_similarity > 0.6 and confirmation_score > contradiction_score and uncertainty_score < 3:  # Lowered from 0.7
            verdict = "TRUE"
            confidence_level = "HIGH" if confidence > 0.65 else "MEDIUM"  # More generous confidence
            reasoning_parts.append(f"High average similarity ({avg_similarity:.3f})")
            reasoning_parts.append(f"More confirmations than contradictions ({confirmation_score} vs {contradiction_score})")
            
        elif avg_similarity > 0.5 and confirmation_score > 0 and contradiction_score == 0:  # New: Support claims with no contradictions
            verdict = "TRUE"
            confidence_level = "MEDIUM" if confidence > 0.5 else "LOW"
            reasoning_parts.append(f"Good similarity ({avg_similarity:.3f}) with confirmation signals and no contradictions")
            
        elif contradiction_score > confirmation_score * 1.2 or (avg_similarity > 0.5 and contradiction_score > 3):  # Lowered from 1.5
            verdict = "FALSE"
            confidence_level = "HIGH" if contradiction_score > 5 else "MEDIUM"
            reasoning_parts.append(f"Strong contradiction signals ({contradiction_score} vs {confirmation_score} confirmations)")
            if avg_similarity > 0.5:
                reasoning_parts.append(f"Good topic match but contradictory content ({avg_similarity:.3f} similarity)")
            
        elif uncertainty_score > confirmation_score + contradiction_score or avg_similarity < 0.3:  # Lowered from 0.4
            verdict = "UNVERIFIABLE"
            confidence_level = "LOW"
            if uncertainty_score > confirmation_score + contradiction_score:
                reasoning_parts.append(f"High uncertainty signals ({uncertainty_score}) suggest unverifiable claim")
            else:
                reasoning_parts.append(f"Low similarity ({avg_similarity:.3f}) with available evidence")
            
        else:
            verdict = "UNVERIFIABLE"
            confidence_level = "LOW"
            reasoning_parts.append(f"Mixed or insufficient evidence for clear determination")
            reasoning_parts.append(f"Similarity: {avg_similarity:.3f}, Confirmations: {confirmation_score}, Contradictions: {contradiction_score}")
        
        # Add source information to reasoning
        if unique_sources > 1:
            reasoning_parts.append(f"Evidence from {unique_sources} different sources")
        
        # Special handling for numerical/date claims - make less strict
        if claim_has_numbers or claim_has_dates:
            reasoning_parts.append("Claim contains specific numbers/dates - requires careful verification")
            if verdict == "TRUE" and max_similarity < 0.8:  # Lowered from 0.9
                verdict = "UNVERIFIABLE"
                confidence_level = "LOW"
                reasoning_parts.append("Downgraded due to insufficient precision for factual claim")
        
        # Final confidence adjustment - more generous mapping
        confidence_mapping = {"HIGH": 0.85, "MEDIUM": 0.7, "LOW": 0.45}  # Increased from 0.8, 0.6, 0.4
        final_confidence = confidence_mapping[confidence_level]
        
        # Boost confidence based on evidence quality
        if avg_similarity > 0.7:
            final_confidence = min(final_confidence + 0.1, 1.0)
        elif avg_similarity > 0.5:
            final_confidence = min(final_confidence + 0.05, 1.0)
        
        # Combine reasoning
        reasoning = f"Rule-based analysis: {' | '.join(reasoning_parts)}"
        
        return {
            "verdict": self.verdicts[verdict],
            "raw_verdict": verdict,
            "confidence": final_confidence,
            "evidence": [fact.get('content', '') for fact in evidence_facts[:3]],
            "evidence_sources": [fact.get('source', 'Unknown') for fact in evidence_facts[:3]],
            "reasoning": reasoning,
            "similarity_scores": [fact.get('similarity_score', 0) for fact in evidence_facts],
            "confidence_level": confidence_level,
            "analysis": f"Processed {len(evidence_facts)} evidence items with avg similarity {avg_similarity:.3f}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_unverifiable_result(self, claim: str, reason: str) -> Dict:
        """Create an unverifiable result"""
        return {
            "verdict": self.verdicts["UNVERIFIABLE"],
            "raw_verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence": [],
            "evidence_sources": [],
            "reasoning": reason,
            "similarity_scores": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def batch_check_claims(self, claims: List[str], evidence_lists: List[List[Dict]]) -> List[Dict]:
        """
        Batch fact-check multiple claims
        
        Args:
            claims: List of claims to check
            evidence_lists: List of evidence lists, one for each claim
            
        Returns:
            List of fact-check results
        """
        results = []
        for claim, evidence in zip(claims, evidence_lists):
            result = self.check_claim(claim, evidence)
            results.append(result)
        
        return results
    
    def get_verdict_summary(self, results: List[Dict]) -> Dict:
        """Get summary statistics for a batch of fact-check results"""
        if not results:
            return {}
        
        verdicts = [result.get('raw_verdict', 'UNVERIFIABLE') for result in results]
        
        summary = {
            'total_claims': len(results),
            'true_count': verdicts.count('TRUE'),
            'false_count': verdicts.count('FALSE'),
            'unverifiable_count': verdicts.count('UNVERIFIABLE'),
            'average_confidence': sum(result.get('confidence', 0) for result in results) / len(results)
        }
        
        summary['true_percentage'] = (summary['true_count'] / summary['total_claims']) * 100
        summary['false_percentage'] = (summary['false_count'] / summary['total_claims']) * 100
        summary['unverifiable_percentage'] = (summary['unverifiable_count'] / summary['total_claims']) * 100
        
        return summary 