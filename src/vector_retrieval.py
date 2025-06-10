import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorRetrieval:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_file: str = "data/faiss_index.bin"):
        """
        Initialize the enhanced vector retrieval system
        
        Args:
            model_name: Name of the sentence transformer model to use
            index_file: Path to save/load the FAISS index
        """
        self.model_name = model_name
        self.index_file = index_file
        self.metadata_file = index_file.replace('.bin', '_metadata.pkl')
        
        # Initialize sentence transformer
        print(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index with better parameters
        # Use both inner product and L2 distance for different similarity measures
        self.index_ip = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index_l2 = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance for Euclidean
        
        # Initialize TF-IDF for keyword-based similarity as backup
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        
        self.fact_metadata = []  # Store original fact data
        self.fact_texts = []     # Store processed fact texts
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        # Load existing index if available
        self.load_index()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation that affects meaning
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"&%$]', ' ', text)
        
        # Normalize some common abbreviations
        abbreviations = {
            ' govt ': ' government ',
            ' pct ': ' percent ',
            ' mil ': ' million ',
            ' bil ': ' billion ',
            ' yr ': ' year ',
            ' yrs ': ' years ',
        }
        
        text_lower = text.lower()
        for abbr, full in abbreviations.items():
            text_lower = text_lower.replace(abbr, full)
        
        return text_lower

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using the sentence transformer with preprocessing
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Generate embeddings with better parameters
        embeddings = self.encoder.encode(
            processed_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        return embeddings
    
    def build_index(self, facts: List[Dict]):
        """
        Build enhanced FAISS indexes and TF-IDF matrix from facts
        
        Args:
            facts: List of fact dictionaries containing 'content' field
        """
        print(f"Building enhanced indexes from {len(facts)} facts...")
        
        # Extract and process text content
        fact_texts = []
        self.fact_metadata = []
        
        for fact in facts:
            content = fact.get('content', '')
            if content.strip():
                processed_content = self._preprocess_text(content)
                fact_texts.append(processed_content)
                self.fact_metadata.append(fact)
        
        if not fact_texts:
            print("No valid facts to index")
            return
        
        self.fact_texts = fact_texts
        
        # Generate embeddings
        embeddings = self.embed_texts([fact.get('content', '') for fact in self.fact_metadata])
        
        # Clear existing indexes and rebuild
        self.index_ip = faiss.IndexFlatIP(self.embedding_dim)
        self.index_l2 = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to both indexes
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)  # Normalize for cosine similarity
        
        self.index_ip.add(embeddings_normalized.astype(np.float32))
        self.index_l2.add(embeddings.astype(np.float32))
        
        # Build TF-IDF matrix for keyword-based fallback
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(fact_texts)
            print(f"TF-IDF matrix built with {self.tfidf_matrix.shape[1]} features")
        except Exception as e:
            print(f"Warning: Could not build TF-IDF matrix: {e}")
            self.tfidf_matrix = None
        
        print(f"Enhanced indexes built with {self.index_ip.ntotal} facts")
        
        # Save indexes and metadata
        self.save_index()
    
    def add_facts(self, facts: List[Dict]):
        """
        Add new facts to the existing indexes
        
        Args:
            facts: List of fact dictionaries to add
        """
        fact_texts = []
        new_metadata = []
        
        for fact in facts:
            content = fact.get('content', '')
            if content.strip():
                processed_content = self._preprocess_text(content)
                fact_texts.append(processed_content)
                new_metadata.append(fact)
        
        if not fact_texts:
            return
        
        # Generate embeddings for new facts
        embeddings = self.embed_texts([fact.get('content', '') for fact in new_metadata])
        
        # Normalize for cosine similarity
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        
        # Add to indexes
        self.index_ip.add(embeddings_normalized.astype(np.float32))
        self.index_l2.add(embeddings.astype(np.float32))
        
        # Add to metadata and texts
        self.fact_metadata.extend(new_metadata)
        self.fact_texts.extend(fact_texts)
        
        # Rebuild TF-IDF matrix with all texts
        if self.fact_texts:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.fact_texts)
            except:
                pass
        
        print(f"Added {len(fact_texts)} facts to indexes. Total: {self.index_ip.ntotal}")
        
        # Save updated indexes
        self.save_index()
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced multi-modal search combining semantic and keyword approaches"""
        if self.index_ip is None or len(self.fact_metadata) == 0:
            return []
        
        # Preprocess query for better matching
        processed_query = self._preprocess_query(query)
        
        # Perform multi-layered search
        semantic_results = self._enhanced_semantic_search(processed_query, k * 2)
        keyword_results = self._enhanced_keyword_search(processed_query, k)
        
        # Combine and rerank results with advanced scoring
        combined_results = self._advanced_rerank_results(
            semantic_results, keyword_results, query, k
        )
        
        # Apply quality filtering and add contextual information
        enhanced_results = self._enhance_results_with_context(combined_results, query)
        
        return enhanced_results[:k]
    
    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing for better matching"""
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations
        abbreviations = {
            'govt': 'government',
            'pres': 'president', 
            'min': 'minister',
            'sec': 'secretary',
            'dept': 'department',
            'corp': 'corporation',
            'inc': 'incorporated',
            'ltd': 'limited'
        }
        
        for abbrev, full_form in abbreviations.items():
            query = re.sub(r'\b' + abbrev + r'\b', full_form, query, flags=re.IGNORECASE)
        
        # Extract and emphasize key entities
        query_entities = self._extract_query_entities(query)
        if query_entities:
            # Boost entity importance in the query
            for entity in query_entities[:3]:  # Limit to avoid query explosion
                if len(entity) > 2:  # Skip very short entities
                    query += f" {entity}"
        
        return query
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract key entities from query for enhanced matching"""
        # Simple entity extraction based on patterns
        entities = []
        
        # Capitalized sequences (likely proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized_words)
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|k|million|billion)?\b', query)
        entities.extend(numbers)
        
        # Years
        years = re.findall(r'\b(?:19|20)\d{2}\b', query)
        entities.extend(years)
        
        return list(set(entities))  # Remove duplicates
    
    def _enhanced_semantic_search(self, query: str, k: int) -> List[Dict]:
        """Enhanced semantic search with multiple similarity measures"""
        # Embed the query with error handling
        try:
            query_embedding = self.embed_texts([query])
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []
        
        if query_embedding.size == 0:
            return []
        
        # Normalize query embedding for cosine similarity
        query_normalized = query_embedding.copy()
        faiss.normalize_L2(query_normalized)
        
        # Multi-strategy search
        results = []
        
        # Strategy 1: Cosine similarity (primary)
        try:
            scores_ip, indices_ip = self.index_ip.search(query_normalized.astype(np.float32), k)
            for score, idx in zip(scores_ip[0], indices_ip[0]):
                if idx < len(self.fact_metadata) and score > 0.1:  # Basic threshold
                    result = self.fact_metadata[idx].copy()
                    result['cosine_similarity'] = float(score)
                    result['semantic_score'] = float(score)
                    result['search_method'] = 'cosine_semantic'
                    results.append(result)
        except Exception as e:
            print(f"Cosine similarity search failed: {e}")
        
        # Strategy 2: L2 distance (secondary validation)
        try:
            scores_l2, indices_l2 = self.index_l2.search(query_embedding.astype(np.float32), k // 2)
            for score, idx in zip(scores_l2[0], indices_l2[0]):
                if idx < len(self.fact_metadata):
                    # Convert L2 distance to similarity
                    l2_similarity = 1.0 / (1.0 + score)
                    if l2_similarity > 0.3:  # Threshold for L2
                        # Check if not already added by cosine search
                        if not any(r.get('original_index') == idx for r in results):
                            result = self.fact_metadata[idx].copy()
                            result['l2_similarity'] = float(l2_similarity)
                            result['semantic_score'] = float(l2_similarity) * 0.8  # Lower weight
                            result['search_method'] = 'l2_semantic'
                            result['original_index'] = idx
                            results.append(result)
        except Exception as e:
            print(f"L2 distance search failed: {e}")
        
        return results
    
    def _enhanced_keyword_search(self, query: str, k: int) -> List[Dict]:
        """Enhanced keyword-based search with TF-IDF and fuzzy matching"""
        if not hasattr(self, 'tfidf_vectorizer') or self.tfidf_vectorizer is None:
            return []
        
        try:
            # Transform query using TF-IDF
            query_tfidf = self.tfidf_vectorizer.transform([query])
            
            if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
                # Calculate TF-IDF similarities
                tfidf_similarities = (self.tfidf_matrix * query_tfidf.T).toarray().flatten()
                
                # Get top matches
                top_indices = np.argsort(tfidf_similarities)[::-1][:k * 2]
                
                results = []
                for idx in top_indices:
                    if idx < len(self.fact_metadata) and tfidf_similarities[idx] > 0.05:
                        result = self.fact_metadata[idx].copy()
                        result['tfidf_similarity'] = float(tfidf_similarities[idx])
                        result['keyword_score'] = float(tfidf_similarities[idx])
                        result['search_method'] = 'tfidf_keyword'
                        
                        # Add fuzzy matching score
                        content = result.get('content', '')
                        fuzzy_score = self._calculate_fuzzy_similarity(query, content)
                        result['fuzzy_similarity'] = fuzzy_score
                        
                        # Combine keyword scores
                        combined_keyword_score = (
                            result['tfidf_similarity'] * 0.7 + 
                            fuzzy_score * 0.3
                        )
                        result['keyword_score'] = combined_keyword_score
                        
                        results.append(result)
                
                return results
        except Exception as e:
            print(f"Enhanced keyword search failed: {e}")
        
        return []
    
    def _calculate_fuzzy_similarity(self, query: str, content: str) -> float:
        """Calculate fuzzy string similarity for better keyword matching"""
        # Simple word-based similarity
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        content_words = set(word.lower() for word in content.split() if len(word) > 2)
        
        if not query_words or not content_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Partial match bonus (for substring matches)
        partial_matches = 0
        for query_word in query_words:
            for content_word in content_words:
                if query_word in content_word or content_word in query_word:
                    if abs(len(query_word) - len(content_word)) <= 2:  # Similar length
                        partial_matches += 1
                        break
        
        partial_bonus = min(partial_matches / len(query_words), 0.5)
        
        return min(jaccard_similarity + partial_bonus, 1.0)
    
    def _advanced_rerank_results(self, semantic_results: List[Dict], keyword_results: List[Dict], query: str, k: int) -> List[Dict]:
        """Advanced result reranking with multi-factor scoring"""
        # Combine results from different methods
        all_results = {}
        
        # Process semantic results
        for result in semantic_results:
            key = self._get_result_key(result)
            if key not in all_results:
                all_results[key] = result.copy()
                all_results[key]['combined_sources'] = ['semantic']
            else:
                # Merge scores if result exists
                existing = all_results[key]
                existing['semantic_score'] = max(
                    existing.get('semantic_score', 0),
                    result.get('semantic_score', 0)
                )
                if 'semantic' not in existing['combined_sources']:
                    existing['combined_sources'].append('semantic')
        
        # Process keyword results
        for result in keyword_results:
            key = self._get_result_key(result)
            if key not in all_results:
                all_results[key] = result.copy()
                all_results[key]['combined_sources'] = ['keyword']
            else:
                # Merge scores
                existing = all_results[key]
                existing['keyword_score'] = max(
                    existing.get('keyword_score', 0),
                    result.get('keyword_score', 0)
                )
                if 'keyword' not in existing['combined_sources']:
                    existing['combined_sources'].append('keyword')
        
        # Calculate enhanced similarity scores
        enhanced_results = []
        for result in all_results.values():
            enhanced_score = self._calculate_enhanced_similarity(result, query)
            result['similarity_score'] = enhanced_score
            result['enhanced_similarity'] = enhanced_score
            enhanced_results.append(result)
        
        # Sort by enhanced similarity score
        enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Apply quality-based filtering
        filtered_results = self._apply_quality_filtering(enhanced_results, query)
        
        return filtered_results
    
    def _get_result_key(self, result: Dict) -> str:
        """Generate unique key for result deduplication"""
        content = result.get('content', '')
        source = result.get('source', '')
        # Use first 100 characters of content + source as key
        return f"{source}:{content[:100]}"
    
    def _calculate_enhanced_similarity(self, result: Dict, query: str) -> float:
        """Calculate enhanced similarity score combining multiple factors"""
        # Base scores
        semantic_score = result.get('semantic_score', 0)
        keyword_score = result.get('keyword_score', 0)
        
        # Multi-source bonus
        sources = result.get('combined_sources', ['unknown'])
        multi_source_bonus = 0.1 if len(sources) > 1 else 0
        
        # Content quality factors
        content = result.get('content', '')
        content_quality = self._assess_content_quality(content, query)
        
        # Recency factor (if available)
        recency_factor = self._calculate_recency_factor(result)
        
        # Source reliability factor
        source_reliability = self._assess_source_reliability(result.get('source', ''))
        
        # Combine all factors with weights
        enhanced_similarity = (
            semantic_score * 0.4 +           # Primary semantic understanding
            keyword_score * 0.25 +           # Keyword relevance
            content_quality * 0.15 +         # Content quality and length
            recency_factor * 0.1 +           # How recent the information is
            source_reliability * 0.05 +      # Source credibility
            multi_source_bonus               # Bonus for multiple search methods
        )
        
        return min(enhanced_similarity, 1.0)
    
    def _assess_content_quality(self, content: str, query: str) -> float:
        """Assess the quality of content for the given query"""
        if not content:
            return 0.0
        
        quality_factors = []
        
        # Length factor (prefer substantial content)
        length_score = min(len(content.split()) / 50, 1.0)
        quality_factors.append(length_score)
        
        # Entity overlap factor
        query_entities = self._extract_query_entities(query)
        if query_entities:
            entity_matches = sum(1 for entity in query_entities if entity.lower() in content.lower())
            entity_score = entity_matches / len(query_entities)
            quality_factors.append(entity_score)
        
        # Information density (presence of factual indicators)
        factual_indicators = ['announced', 'confirmed', 'reported', 'stated', 'according to', 'data shows']
        indicator_score = sum(1 for indicator in factual_indicators if indicator in content.lower()) / len(factual_indicators)
        quality_factors.append(indicator_score * 0.5)  # Lower weight
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _calculate_recency_factor(self, result: Dict) -> float:
        """Calculate recency factor based on publication date"""
        # For now, return neutral score since date parsing is complex
        # This could be enhanced with proper date parsing
        published_date = result.get('published_date', '')
        if published_date and published_date != 'Unknown':
            try:
                # Simple check for recent years
                if any(year in published_date for year in ['2024', '2023', '2022']):
                    return 0.8
                elif any(year in published_date for year in ['2021', '2020']):
                    return 0.6
                else:
                    return 0.4
            except:
                pass
        return 0.5  # Neutral score for unknown dates
    
    def _assess_source_reliability(self, source: str) -> float:
        """Assess source reliability based on known patterns"""
        if not source or source == 'Unknown':
            return 0.4
        
        source_lower = source.lower()
        
        # High reliability sources
        high_reliability = [
            'bbc', 'reuters', 'associated press', 'npr', 'guardian',
            'bloomberg', 'wall street journal', 'financial times',
            'nasa', 'who', 'government', 'official'
        ]
        
        # Medium reliability sources
        medium_reliability = [
            'cnn', 'nbc', 'abc', 'times', 'post', 'news',
            'techcrunch', 'ars technica', 'wired'
        ]
        
        for high_source in high_reliability:
            if high_source in source_lower:
                return 0.9
        
        for medium_source in medium_reliability:
            if medium_source in source_lower:
                return 0.7
        
        return 0.5  # Default reliability score
    
    def _apply_quality_filtering(self, results: List[Dict], query: str) -> List[Dict]:
        """Apply quality-based filtering to results"""
        if not results:
            return []
        
        # Calculate quality threshold based on best result
        best_score = max(result['similarity_score'] for result in results)
        quality_threshold = max(best_score * 0.3, 0.15)  # Adaptive threshold
        
        # Filter results
        filtered_results = []
        for result in results:
            if result['similarity_score'] >= quality_threshold:
                # Add quality indicators
                result['quality_indicators'] = self._get_quality_indicators(result, query)
                filtered_results.append(result)
        
        return filtered_results
    
    def _get_quality_indicators(self, result: Dict, query: str) -> List[str]:
        """Get quality indicators for a result"""
        indicators = []
        
        # High similarity indicator
        if result['similarity_score'] > 0.7:
            indicators.append('high_similarity')
        
        # Multi-method confirmation
        if len(result.get('combined_sources', [])) > 1:
            indicators.append('multi_method_match')
        
        # Source reliability
        source_reliability = self._assess_source_reliability(result.get('source', ''))
        if source_reliability > 0.8:
            indicators.append('reliable_source')
        
        # Content richness
        content_length = len(result.get('content', '').split())
        if content_length > 30:
            indicators.append('detailed_content')
        
        # Entity matching
        query_entities = self._extract_query_entities(query)
        if query_entities:
            content = result.get('content', '').lower()
            entity_matches = sum(1 for entity in query_entities if entity.lower() in content)
            if entity_matches >= len(query_entities) * 0.7:
                indicators.append('strong_entity_match')
        
        return indicators
    
    def _enhance_results_with_context(self, results: List[Dict], query: str) -> List[Dict]:
        """Enhance results with additional context and metadata"""
        enhanced_results = []
        
        for result in results:
            # Add query-specific relevance explanation
            result['relevance_explanation'] = self._generate_detailed_relevance_explanation(query, result)
            
            # Add confidence indicators
            result['confidence_indicators'] = self._calculate_confidence_indicators(result)
            
            # Add fact-checking context
            result['fact_check_context'] = self._generate_fact_check_context(result, query)
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _generate_detailed_relevance_explanation(self, query: str, result: Dict) -> str:
        """Generate detailed explanation of relevance"""
        explanations = []
        
        # Similarity explanation
        similarity = result.get('similarity_score', 0)
        if similarity > 0.8:
            explanations.append(f"Very high relevance ({similarity:.2f}) - strong semantic match")
        elif similarity > 0.6:
            explanations.append(f"High relevance ({similarity:.2f}) - good semantic alignment")
        elif similarity > 0.4:
            explanations.append(f"Moderate relevance ({similarity:.2f}) - partial content match")
        else:
            explanations.append(f"Limited relevance ({similarity:.2f}) - weak content connection")
        
        # Method explanation
        sources = result.get('combined_sources', [])
        if len(sources) > 1:
            explanations.append(f"Confirmed by multiple search methods: {', '.join(sources)}")
        
        # Entity matching
        query_entities = self._extract_query_entities(query)
        if query_entities:
            content = result.get('content', '').lower()
            matched_entities = [entity for entity in query_entities if entity.lower() in content]
            if matched_entities:
                explanations.append(f"Matches key entities: {', '.join(matched_entities[:3])}")
        
        return ' | '.join(explanations)
    
    def _calculate_confidence_indicators(self, result: Dict) -> Dict:
        """Calculate confidence indicators for the result"""
        indicators = {
            'similarity_confidence': min(result.get('similarity_score', 0) * 1.2, 1.0),
            'source_confidence': self._assess_source_reliability(result.get('source', '')),
            'content_confidence': min(len(result.get('content', '').split()) / 50, 1.0),
            'method_confidence': len(result.get('combined_sources', [])) / 2.0
        }
        
        # Overall confidence
        indicators['overall_confidence'] = sum(indicators.values()) / len(indicators)
        
        return indicators
    
    def _generate_fact_check_context(self, result: Dict, query: str) -> Dict:
        """Generate context specifically for fact-checking"""
        content = result.get('content', '')
        
        context = {
            'content_type': self._classify_content_type(content),
            'factual_indicators': self._identify_factual_indicators(content),
            'verification_signals': self._identify_verification_signals(content),
            'temporal_context': self._extract_temporal_context(content),
            'numerical_context': self._extract_numerical_context(content)
        }
        
        return context
    
    def _identify_factual_indicators(self, content: str) -> List[str]:
        """Identify indicators that suggest factual content"""
        indicators = []
        content_lower = content.lower()
        
        factual_patterns = [
            ('official_statement', ['announced', 'confirmed', 'stated officially', 'declared']),
            ('data_reference', ['according to data', 'statistics show', 'research indicates']),
            ('attribution', ['according to', 'reported by', 'as stated by']),
            ('verification', ['verified', 'confirmed', 'validated', 'authenticated']),
            ('quantitative', [r'\d+%', r'\$\d+', r'\d+ million', r'\d+ billion']),
        ]
        
        for indicator_type, patterns in factual_patterns:
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    indicators.append(indicator_type)
                    break
        
        return indicators
    
    def _identify_verification_signals(self, content: str) -> List[str]:
        """Identify signals that help with verification"""
        signals = []
        content_lower = content.lower()
        
        verification_patterns = {
            'contradiction': ['denied', 'contradicts', 'disputes', 'refutes'],
            'confirmation': ['confirms', 'supports', 'validates', 'corroborates'],
            'uncertainty': ['alleged', 'reportedly', 'claimed', 'unconfirmed'],
            'official': ['government', 'ministry', 'department', 'official'],
            'recent': ['today', 'yesterday', 'this week', 'recently']
        }
        
        for signal_type, keywords in verification_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                signals.append(signal_type)
        
        return signals
    
    def _extract_temporal_context(self, content: str) -> List[str]:
        """Extract temporal context from content"""
        temporal_elements = []
        
        # Years
        years = re.findall(r'\b(?:19|20)\d{2}\b', content)
        temporal_elements.extend(years)
        
        # Months
        months = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', content.lower())
        temporal_elements.extend(months)
        
        # Relative time
        relative_time = re.findall(r'\b(?:today|yesterday|tomorrow|this week|last week|this month|last month|this year|last year)\b', content.lower())
        temporal_elements.extend(relative_time)
        
        return list(set(temporal_elements))
    
    def _extract_numerical_context(self, content: str) -> List[str]:
        """Extract numerical context from content"""
        numerical_elements = []
        
        # Percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', content)
        numerical_elements.extend(percentages)
        
        # Money amounts
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', content)
        numerical_elements.extend(money)
        
        # Large numbers
        large_numbers = re.findall(r'\d+(?:,\d{3})*(?:\s*(?:million|billion|trillion))', content)
        numerical_elements.extend(large_numbers)
        
        return numerical_elements

    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content for better fact checking"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['announced', 'reported', 'confirmed', 'stated']):
            return 'news_report'
        elif any(word in content_lower for word in ['policy', 'law', 'regulation', 'government']):
            return 'official_statement'
        elif any(word in content_lower for word in ['study', 'research', 'survey', 'analysis']):
            return 'research_finding'
        elif re.search(r'\d+(\.\d+)?%', content) or re.search(r'\$\d+', content):
            return 'statistical_data'
        else:
            return 'general_information'

    def get_similar_facts(self, claim_text: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Get facts similar to a claim with enhanced filtering and context awareness
        
        Args:
            claim_text: The claim to find similar facts for
            top_k: Maximum number of facts to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of similar facts with enhanced metadata
        """
        # Perform enhanced search
        results = self.search(claim_text, k=top_k * 3)  # Get more candidates for filtering
        
        # Apply enhanced filtering
        filtered_results = []
        for result in results:
            similarity_score = result.get('similarity_score', 0)
            
            # Apply threshold
            if similarity_score >= similarity_threshold:
                # Add contextual metadata
                result['retrieval_confidence'] = self._calculate_retrieval_confidence(claim_text, result)
                result['content_type'] = self._classify_content_type(result.get('content', ''))
                result['relevance_explanation'] = self._generate_detailed_relevance_explanation(claim_text, result)
                
                filtered_results.append(result)
        
        # Sort by similarity and return top results
        filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return filtered_results[:top_k]
    
    def _calculate_retrieval_confidence(self, query: str, result: Dict) -> float:
        """Calculate confidence in the retrieval result"""
        factors = []
        
        # Similarity score factor
        similarity = result.get('similarity_score', 0)
        factors.append(min(similarity * 2, 1.0))
        
        # Multiple score agreement factor
        semantic_score = result.get('semantic_score', 0)
        keyword_score = result.get('keyword_score', 0)
        
        if semantic_score > 0 and keyword_score > 0:
            score_agreement = 1.0 - abs(semantic_score - keyword_score)
            factors.append(score_agreement * 0.5)
        
        # Content quality factor
        content = result.get('content', '')
        quality_score = min(len(content.split()) / 50, 1.0)  # Prefer longer content
        factors.append(quality_score * 0.3)
        
        return min(sum(factors) / len(factors), 1.0)

    def save_index(self):
        """Save the FAISS indexes and metadata to files"""
        try:
            # Save FAISS indexes
            faiss.write_index(self.index_ip, self.index_file)
            faiss.write_index(self.index_l2, self.index_file.replace('.bin', '_l2.bin'))
            
            # Save metadata and additional data
            with open(self.metadata_file, 'wb') as f:
                data_to_save = {
                    'fact_metadata': self.fact_metadata,
                    'fact_texts': self.fact_texts,
                    'tfidf_vectorizer': self.tfidf_vectorizer if hasattr(self.tfidf_vectorizer, 'vocabulary_') else None
                }
                pickle.dump(data_to_save, f)
            
            print(f"Enhanced indexes saved to {self.index_file}")
            
        except Exception as e:
            print(f"Error saving indexes: {e}")
    
    def load_index(self):
        """Load the FAISS indexes and metadata from files"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS indexes
                self.index_ip = faiss.read_index(self.index_file)
                
                l2_file = self.index_file.replace('.bin', '_l2.bin')
                if os.path.exists(l2_file):
                    self.index_l2 = faiss.read_index(l2_file)
                else:
                    self.index_l2 = faiss.IndexFlatL2(self.embedding_dim)
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.fact_metadata = data.get('fact_metadata', [])
                        self.fact_texts = data.get('fact_texts', [])
                        saved_vectorizer = data.get('tfidf_vectorizer')
                        if saved_vectorizer:
                            self.tfidf_vectorizer = saved_vectorizer
                    else:
                        # Backward compatibility
                        self.fact_metadata = data
                        self.fact_texts = []
                
                # Rebuild TF-IDF if needed
                if not hasattr(self.tfidf_vectorizer, 'vocabulary_') and self.fact_texts:
                    try:
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.fact_texts)
                    except:
                        pass
                
                print(f"Enhanced indexes loaded with {self.index_ip.ntotal} facts")
                
        except Exception as e:
            print(f"Could not load existing indexes: {e}")
            # Initialize empty indexes
            self.index_ip = faiss.IndexFlatIP(self.embedding_dim)
            self.index_l2 = faiss.IndexFlatL2(self.embedding_dim)
            self.fact_metadata = []
            self.fact_texts = []

    def get_index_stats(self) -> Dict:
        """Get detailed statistics about the indexes"""
        stats = {
            'total_facts': self.index_ip.ntotal,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'has_tfidf': self.tfidf_matrix is not None,
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'index_types': ['cosine_similarity', 'l2_distance', 'tfidf_keyword']
        }
        
        if self.fact_metadata:
            # Analyze content types
            content_types = {}
            for fact in self.fact_metadata:
                content_type = self._classify_content_type(fact.get('content', ''))
                content_types[content_type] = content_types.get(content_type, 0) + 1
            stats['content_distribution'] = content_types
        
        return stats

    def clear_index(self):
        """Clear all indexes and metadata"""
        self.index_ip = faiss.IndexFlatIP(self.embedding_dim)
        self.index_l2 = faiss.IndexFlatL2(self.embedding_dim)
        self.fact_metadata = []
        self.fact_texts = []
        self.tfidf_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        print("All indexes cleared") 