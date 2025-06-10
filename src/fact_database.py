import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict
from datetime import datetime, timedelta
import os

class FactDatabase:
    def __init__(self, cache_file: str = "data/facts_cache.json"):
        self.cache_file = cache_file
        self.facts = []
        
        # Enhanced RSS feeds - updated based on testing results for maximum reliability
        self.rss_feeds = [
            # International News Sources (Working feeds)
            {
                'url': "http://feeds.bbci.co.uk/news/world/rss.xml",
                'name': "BBC World News",
                'max_entries': 50
            },
            {
                'url': "https://www.theguardian.com/world/rss",
                'name': "The Guardian World",
                'max_entries': 50
            },
            {
                'url': "https://www.npr.org/rss/rss.php?id=1001",
                'name': "NPR News",
                'max_entries': 30
            },
            {
                'url': "https://feeds.nbcnews.com/nbcnews/public/world",
                'name': "NBC World News",
                'max_entries': 30
            },
            {
                'url': "https://feeds.skynews.com/feeds/rss/world.xml",
                'name': "Sky News World",
                'max_entries': 40
            },
            
            # Indian News Sources
            {
                'url': "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
                'name': "Times of India",
                'max_entries': 40
            },
            {
                'url': "https://indianexpress.com/section/india/feed/",
                'name': "Indian Express",
                'max_entries': 40
            },
            {
                'url': "https://www.business-standard.com/rss/home_page_top_stories.rss",
                'name': "Business Standard India",
                'max_entries': 30
            },
            
            # Technology Sources
            {
                'url': "http://feeds.arstechnica.com/arstechnica/index",
                'name': "Ars Technica",
                'max_entries': 30
            },
            {
                'url': "https://techcrunch.com/feed/",
                'name': "TechCrunch",
                'max_entries': 30
            },
            {
                'url': "https://feeds.feedburner.com/venturebeat/SZYF",
                'name': "VentureBeat",
                'max_entries': 25
            },
            {
                'url': "https://www.wired.com/feed/rss",
                'name': "Wired",
                'max_entries': 25
            },
            
            # Business & Finance Sources (Working feeds)
            {
                'url': "https://feeds.bloomberg.com/markets/news.rss",
                'name': "Bloomberg Markets",
                'max_entries': 40
            },
            {
                'url': "https://feeds.marketwatch.com/marketwatch/topstories/",
                'name': "MarketWatch",
                'max_entries': 30
            },
            
            # Science & Health Sources
            {
                'url': "https://www.sciencedaily.com/rss/all.xml",
                'name': "Science Daily",
                'max_entries': 30
            },
            {
                'url': "https://feeds.feedburner.com/reuters/scienceNews",
                'name': "Reuters Science",
                'max_entries': 25
            },
            {
                'url': "https://feeds.feedburner.com/reuters/healthNews",
                'name': "Reuters Health",
                'max_entries': 25
            },
            
            # Government & Official Sources
            {
                'url': "https://www.nasa.gov/rss/dyn/breaking_news.rss",
                'name': "NASA News",
                'max_entries': 20
            },
            {
                'url': "https://www.who.int/rss-feeds/news-english.xml",
                'name': "World Health Organization",
                'max_entries': 20
            },
            
            # Additional Reliable Sources
            {
                'url': "https://feeds.feedburner.com/time/topstories",
                'name': "TIME Magazine",
                'max_entries': 25
            },
            {
                'url': "https://feeds.feedburner.com/newsweek",
                'name': "Newsweek",
                'max_entries': 25
            },
            {
                'url': "https://www.economist.com/latest/rss.xml",
                'name': "The Economist",
                'max_entries': 25
            }
        ]
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Load cached facts
        self.load_cached_facts()
    
    def load_cached_facts(self):
        """Load facts from cache file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.facts = data.get('facts', [])
                    print(f"Loaded {len(self.facts)} facts from cache")
            except Exception as e:
                print(f"Error loading cached facts: {e}")
                self.facts = []
    
    def save_facts_to_cache(self):
        """Save facts to cache file"""
        try:
            data = {
                'facts': self.facts,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.facts)} facts to cache")
        except Exception as e:
            print(f"Error saving facts to cache: {e}")
    
    def fetch_from_rss_feeds(self, max_total_facts: int = 500) -> List[Dict]:
        """Fetch recent news from RSS feeds - now capable of fetching hundreds of facts"""
        all_entries = []
        successful_feeds = 0
        failed_feeds = 0
        
        print(f"Starting RSS retrieval from {len(self.rss_feeds)} sources...")
        print(f"Target: {max_total_facts} total facts")
        
        for i, feed_config in enumerate(self.rss_feeds, 1):
            feed_url = feed_config['url']
            feed_name = feed_config['name']
            max_entries = feed_config['max_entries']
            
            try:
                print(f"[{i}/{len(self.rss_feeds)}] Fetching from {feed_name}...")
                print(f"  URL: {feed_url}")
                
                # Parse the RSS feed
                feed = feedparser.parse(feed_url)
                
                # Check if feed was parsed successfully
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    print(f"  ❌ No entries found or feed parsing failed")
                    failed_feeds += 1
                    continue
                
                entries_processed = 0
                entries_added = 0
                
                # Process entries from this feed
                for entry in feed.entries[:max_entries]:
                    entries_processed += 1
                    
                    try:
                        # Extract title - this is crucial for fact retrieval
                        title = entry.get('title', '').strip()
                        if not title:
                            continue
                        
                        # Extract and clean the content
                        content = self._extract_content(entry)
                        
                        # Create comprehensive content combining title and description
                        full_content = self._create_comprehensive_content(title, content, entry)
                        
                        # Check if this is factual content worth storing
                        if self._is_factual_content(full_content, title):
                            fact_entry = {
                                'title': title,
                                'content': full_content,
                                'source': feed_name,
                                'url': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'published_parsed': entry.get('published_parsed', None),
                                'summary': entry.get('summary', '')[:200] + '...' if entry.get('summary') else '',
                                'fetched_date': datetime.now().isoformat(),
                                'feed_url': feed_url
                            }
                            all_entries.append(fact_entry)
                            entries_added += 1
                    
                    except Exception as entry_error:
                        print(f"    ⚠️ Error processing entry: {entry_error}")
                        continue
                
                print(f"  ✅ {entries_added}/{entries_processed} entries added from {feed_name}")
                successful_feeds += 1
                
                # Stop if we've reached our target
                if len(all_entries) >= max_total_facts:
                    print(f"  🎯 Reached target of {max_total_facts} facts")
                    break
                
            except Exception as e:
                print(f"  ❌ Error fetching from {feed_name}: {e}")
                failed_feeds += 1
                continue
        
        print(f"\nRSS Retrieval Summary:")
        print(f"  ✅ Successful feeds: {successful_feeds}")
        print(f"  ❌ Failed feeds: {failed_feeds}")
        print(f"  📰 Total facts retrieved: {len(all_entries)}")
        
        return all_entries
    
    def _extract_content(self, entry) -> str:
        """Extract clean text content from RSS entry"""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        elif hasattr(entry, 'content'):
            if isinstance(entry.content, list):
                content = entry.content[0].value
            else:
                content = str(entry.content)
        
        # Clean HTML tags
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text()
            
        # Clean and normalize text
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Combine with title if content is short
        title = entry.get('title', '')
        if len(content) < 100 and title:
            content = f"{title}. {content}"
        
        return content
    
    def _is_factual_content(self, content: str, title: str) -> bool:
        """Enhanced filter for factual content using both content and title"""
        if len(content) < 30:  # Too short
            return False
        
        combined_text = f"{title} {content}".lower()
        
        # Filter out common non-factual patterns
        skip_patterns = [
            r'cookie policy', r'privacy policy', r'terms of service',
            r'subscribe', r'newsletter', r'advertisement', r'sponsored',
            r'follow us', r'share this', r'read more', r'click here',
            r'breaking news alert', r'live blog', r'photo gallery',
            r'watch:', r'video:', r'listen:', r'podcast:',
            r'opinion:', r'editorial:', r'commentary:',
            r'horoscope', r'crossword', r'sudoku',
            r'weather forecast', r'sports scores only',
            r'stock prices', r'market close'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, combined_text):
                return False
        
        # Look for strong factual indicators
        strong_factual_indicators = [
            # Official announcements
            r'announced|declared|confirmed|stated officially|reported',
            r'government|ministry|department|official statement|press release',
            r'president|prime minister|minister|spokesperson said',
            
            # Legal and regulatory
            r'law|legislation|regulation|policy|bill|act|supreme court',
            r'signed into law|passed|approved|ratified',
            
            # Economic and statistical
            r'gdp|inflation|unemployment|economic growth',
            r'\d+(\.\d+)?%|\$\d+|billion|million|crore|lakh',
            r'quarter|fiscal year|annual report|budget',
            
            # Scientific and research
            r'study|research|survey|data shows|findings|published',
            r'university|institute|scientists|researchers',
            
            # International relations
            r'treaty|agreement|summit|diplomatic|embassy',
            r'united nations|nato|european union|g7|g20',
            
            # Technology and business
            r'launched|released|introduced|unveiled|merger|acquisition',
            r'company|corporation|startup|ipo|stock exchange',
            
            # Events and incidents
            r'happened|occurred|took place|incident|event',
            r'died|killed|injured|arrested|rescued',
            r'earthquake|flood|fire|accident|disaster'
        ]
        
        factual_score = 0
        for indicator in strong_factual_indicators:
            if re.search(indicator, combined_text):
                factual_score += 1
        
        # Check for specific dates, numbers, or proper nouns
        has_specific_info = bool(
            re.search(r'\b(19|20)\d{2}\b', combined_text) or  # Years
            re.search(r'\b\d{1,3}(,\d{3})+\b', combined_text) or  # Large numbers
            re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content) or  # Proper names
            re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', combined_text)  # Days
        )
        
        # Decision logic
        if factual_score >= 2:  # Strong factual indicators
            return True
        elif factual_score >= 1 and has_specific_info:  # Good combination
            return True
        elif len(content) > 200 and has_specific_info:  # Long content with specifics
            return True
        elif len(title) > 50 and factual_score >= 1:  # Detailed title with some indicators
            return True
        
        return False
    
    def _create_comprehensive_content(self, title: str, content: str, entry) -> str:
        """Create comprehensive content by combining title, summary, and content"""
        parts = []
        
        # Always start with the title as it's the most important
        if title:
            parts.append(title)
        
        # Add summary if available and different from title
        summary = entry.get('summary', '').strip()
        if summary and summary.lower() != title.lower():
            # Clean HTML from summary
            if summary:
                soup = BeautifulSoup(summary, 'html.parser')
                clean_summary = soup.get_text().strip()
                if clean_summary and len(clean_summary) > 20:
                    parts.append(clean_summary)
        
        # Add main content if it's substantial and different
        if content and len(content) > 50:
            content_lower = content.lower()
            title_lower = title.lower()
            
            # Only add if content is significantly different from title
            if not (content_lower.startswith(title_lower) and len(content) < len(title) * 2):
                parts.append(content)
        
        # Join all parts
        full_content = '. '.join(parts)
        
        # Clean and normalize
        full_content = re.sub(r'\s+', ' ', full_content).strip()
        
        # Ensure it ends with proper punctuation
        if full_content and not full_content.endswith(('.', '!', '?')):
            full_content += '.'
        
        return full_content
    
    def update_database(self, max_facts: int = 500):
        """Update the fact database with fresh content - now scales to hundreds of facts"""
        print("Updating fact database with enhanced RSS retrieval...")
        
        # Fetch new facts from RSS feeds (target hundreds of facts)
        new_facts = self.fetch_from_rss_feeds(max_total_facts=max_facts * 2)  # Fetch more to allow for filtering
        
        # Add manual seed facts if database is empty
        if not self.facts:
            seed_facts = self._get_enhanced_seed_facts()
            self.facts.extend(seed_facts)
            print(f"Added {len(seed_facts)} seed facts")
        
        # Add new facts, avoiding duplicates
        facts_before = len(self.facts)
        duplicates_found = 0
        
        for fact in new_facts:
            if not self._is_duplicate(fact):
                self.facts.append(fact)
            else:
                duplicates_found += 1
        
        facts_after = len(self.facts)
        facts_added = facts_after - facts_before
        
        print(f"Facts processing summary:")
        print(f"  📰 Retrieved from RSS: {len(new_facts)}")
        print(f"  📋 Facts before update: {facts_before}")
        print(f"  ➕ New facts added: {facts_added}")
        print(f"  🔄 Duplicates skipped: {duplicates_found}")
        
        # Sort by date (newest first) and keep the specified maximum
        self.facts = sorted(self.facts, 
                          key=lambda x: x.get('fetched_date', ''), 
                          reverse=True)[:max_facts]
        
        # Save to cache
        self.save_facts_to_cache()
        
        print(f"  💾 Final database size: {len(self.facts)} facts")
        print("Database update completed successfully!")
    
    def _is_duplicate(self, new_fact: Dict) -> bool:
        """Check if a fact is already in the database"""
        new_content = new_fact.get('content', '').lower()
        
        for existing_fact in self.facts:
            existing_content = existing_fact.get('content', '').lower()
            
            # Simple similarity check (could be improved with embeddings)
            if len(new_content) > 0 and len(existing_content) > 0:
                # Check for significant overlap
                words_new = set(new_content.split())
                words_existing = set(existing_content.split())
                
                if len(words_new & words_existing) / len(words_new | words_existing) > 0.7:
                    return True
        
        return False
    
    def _get_enhanced_seed_facts(self) -> List[Dict]:
        """Get comprehensive seed facts covering various domains"""
        seed_facts = [
            # Economic Facts
            {
                'title': 'India GDP Growth Q2 2023-24',
                'content': 'India\'s GDP grew by 7.6% in the second quarter of fiscal 2023-24, according to government statistics released by the Ministry of Statistics.',
                'source': 'Government Statistics',
                'url': '',
                'published': '2023-11-30',
                'fetched_date': datetime.now().isoformat()
            },
            {
                'title': 'India Renewable Energy Target 2030',
                'content': 'India aims to achieve 50% of its electricity needs from renewable sources by 2030, as announced by the Ministry of New and Renewable Energy.',
                'source': 'Ministry of New and Renewable Energy',
                'url': '',
                'published': '2023-11-15',
                'fetched_date': datetime.now().isoformat()
            },
            {
                'title': 'Digital India Aadhaar Statistics',
                'content': 'Over 130 crore Aadhaar cards have been issued in India as part of the Digital India initiative, making it the world\'s largest biometric ID system.',
                'source': 'UIDAI',
                'url': '',
                'published': '2023-10-20',
                'fetched_date': datetime.now().isoformat()
            },
            
            # International Facts
            {
                'title': 'United Nations Climate Summit 2023',
                'content': 'The United Nations Climate Summit 2023 concluded with 195 countries agreeing to transition away from fossil fuels.',
                'source': 'United Nations',
                'url': '',
                'published': '2023-12-13',
                'fetched_date': datetime.now().isoformat()
            },
            {
                'title': 'Global Population Milestone',
                'content': 'The world population reached 8 billion people in late 2022, according to United Nations demographic projections.',
                'source': 'UN Population Division',
                'url': '',
                'published': '2023-01-15',
                'fetched_date': datetime.now().isoformat()
            },
            
            # Technology Facts
            {
                'title': 'Artificial Intelligence Development',
                'content': 'OpenAI released GPT-4 in March 2023, marking a significant advancement in large language model capabilities.',
                'source': 'OpenAI',
                'url': '',
                'published': '2023-03-14',
                'fetched_date': datetime.now().isoformat()
            },
            {
                'title': 'Electric Vehicle Sales Growth',
                'content': 'Global electric vehicle sales increased by 68% in 2023, with over 14 million units sold worldwide.',
                'source': 'International Energy Agency',
                'url': '',
                'published': '2024-01-10',
                'fetched_date': datetime.now().isoformat()
            }
        ]
        return seed_facts
    
    def get_all_facts(self) -> List[Dict]:
        """Get all facts in the database"""
        return self.facts
    
    def get_facts_text(self) -> List[str]:
        """Get just the text content of all facts"""
        return [fact.get('content', '') for fact in self.facts if fact.get('content')]
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text-based search through facts"""
        query_lower = query.lower()
        matching_facts = []
        
        for fact in self.facts:
            content = fact.get('content', '').lower()
            title = fact.get('title', '').lower()
            
            # Simple keyword matching
            if query_lower in content or query_lower in title:
                matching_facts.append(fact)
        
        return matching_facts[:limit]
    
    def get_stats(self) -> Dict:
        """Get statistics about the fact database"""
        if not self.facts:
            return {
                'total_facts': 0,
                'sources': {},
                'recent_facts': 0,
                'avg_content_length': 0
            }
        
        # Count facts by source
        sources = {}
        total_length = 0
        recent_count = 0
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for fact in self.facts:
            # Source count
            source = fact.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
            
            # Content length
            content = fact.get('content', '')
            total_length += len(content)
            
            # Recent facts count
            fetched_date_str = fact.get('fetched_date', '')
            if fetched_date_str:
                try:
                    fetched_date = datetime.fromisoformat(fetched_date_str.replace('Z', '+00:00'))
                    if fetched_date.replace(tzinfo=None) > cutoff_date:
                        recent_count += 1
                except:
                    pass
        
        return {
            'total_facts': len(self.facts),
            'sources': sources,
            'recent_facts': recent_count,
            'avg_content_length': total_length // len(self.facts) if self.facts else 0
        } 