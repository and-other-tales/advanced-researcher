"""Domain detection functionality for routing queries to appropriate knowledge graphs."""
import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from langchain_core.language_models import LanguageModelLike

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    """Enum of available domain types."""
    TAX = "tax"
    LEGAL = "legal"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GENERAL = "general"  # Default fallback domain


# Domain-specific keywords for detection
DOMAIN_KEYWORDS = {
    DomainType.TAX: {
        "tax", "hmrc", "taxation", "taxes", "inland revenue", "self-assessment", "vat", 
        "income tax", "corporation tax", "capital gains", "inheritance tax", "tax return",
        "tax code", "paye", "hmrc.gov.uk", "tax credit", "tax relief", "tax year",
        "tax liability", "tax allowance", "tax deduction", "tax exemption"
    },
    DomainType.LEGAL: {
        "legal", "law", "legislation", "statute", "court", "judge", "lawyer", "solicitor",
        "barrister", "attorney", "jurisdiction", "regulation", "compliance", "contract",
        "lawsuit", "litigation", "legal case", "legislature", "judicial", "legal precedent",
        "legal framework", "legislation.gov.uk", "legal code", "legal rights", "legal obligation"
    },
    DomainType.FINANCE: {
        "finance", "banking", "investment", "stock", "share", "bond", "market", "currency",
        "loan", "mortgage", "interest rate", "credit", "debit", "dividend", "portfolio",
        "financial statement", "balance sheet", "income statement", "cash flow", "asset",
        "liability", "equity", "financial market", "financial instrument", "financial service"
    },
    DomainType.TECHNOLOGY: {
        "technology", "computer", "software", "hardware", "programming", "code", "algorithm",
        "data", "network", "internet", "web", "cloud", "server", "database", "encryption",
        "cybersecurity", "artificial intelligence", "machine learning", "blockchain", "api",
        "coding", "app", "application", "interface", "tech", "tech stack", "saas", "paas"
    },
    DomainType.HEALTHCARE: {
        "healthcare", "medical", "health", "medicine", "doctor", "nurse", "patient", "hospital",
        "clinic", "treatment", "therapy", "diagnosis", "prescription", "pharmaceutical", "drug",
        "surgery", "disease", "condition", "symptom", "vaccination", "immunization", "nhs",
        "healthcare provider", "healthcare system", "public health", "mental health"
    },
    DomainType.EDUCATION: {
        "education", "school", "university", "college", "learning", "teaching", "student",
        "teacher", "professor", "curriculum", "course", "class", "lecture", "study", "research",
        "academic", "degree", "qualification", "scholarship", "education system", "learning outcome",
        "educational institution", "pedagogy", "educational policy", "educational technology"
    }
}

# Map known URLs or sources to their domains
URL_DOMAIN_MAPPING = {
    "hmrc.gov.uk": DomainType.TAX,
    "tax.service.gov.uk": DomainType.TAX,
    "gov.uk/hmrc": DomainType.TAX,
    "gov.uk/tax": DomainType.TAX,
    
    "legislation.gov.uk": DomainType.LEGAL,
    "justice.gov.uk": DomainType.LEGAL,
    "judiciary.uk": DomainType.LEGAL,
    "bailii.org": DomainType.LEGAL,
    
    "bankofengland.co.uk": DomainType.FINANCE,
    "fca.org.uk": DomainType.FINANCE,
    "londonstockexchange.com": DomainType.FINANCE,
    
    "nhs.uk": DomainType.HEALTHCARE,
    "health.org.uk": DomainType.HEALTHCARE,
    "nice.org.uk": DomainType.HEALTHCARE,
    
    "gov.uk/education": DomainType.EDUCATION,
    "ofsted.gov.uk": DomainType.EDUCATION,
    "ucas.com": DomainType.EDUCATION
}

# Map domains to their corresponding Weaviate graph/collection names
DOMAIN_GRAPH_MAPPING = {
    DomainType.TAX: "TaxKnowledge",
    DomainType.LEGAL: "LegalKnowledge",
    DomainType.FINANCE: "FinanceKnowledge",
    DomainType.TECHNOLOGY: "TechnologyKnowledge",
    DomainType.HEALTHCARE: "HealthcareKnowledge",
    DomainType.EDUCATION: "EducationKnowledge",
    DomainType.GENERAL: "GeneralKnowledge"  # Default graph
}


def detect_domain_from_url(url: str) -> Optional[DomainType]:
    """Detect the domain from a URL.
    
    Args:
        url: The URL to analyze
        
    Returns:
        Optional[DomainType]: The detected domain or None if not found
    """
    if not url:
        return None
        
    for url_pattern, domain in URL_DOMAIN_MAPPING.items():
        if url_pattern in url.lower():
            return domain
            
    return None


def detect_domain_from_text(text: str) -> DomainType:
    """Detect the domain from text content.
    
    Args:
        text: The text to analyze
        
    Returns:
        DomainType: The detected domain or GENERAL if no specific domain is detected
    """
    text_lower = text.lower()
    domain_scores = {domain: 0 for domain in DomainType}
    
    # Count domain-specific keywords
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            # Count exact matches
            exact_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            domain_scores[domain] += exact_matches
            
    # Find the domain with the highest score
    max_score = 0
    max_domain = DomainType.GENERAL
    
    for domain, score in domain_scores.items():
        if score > max_score:
            max_score = score
            max_domain = domain
            
    # If the maximum score is low, default to GENERAL
    if max_score < 2:
        return DomainType.GENERAL
        
    return max_domain


def get_graph_name_for_domain(domain: DomainType) -> str:
    """Get the corresponding Weaviate graph name for a domain.
    
    Args:
        domain: The domain to get the graph name for
        
    Returns:
        str: The Weaviate graph/collection name
    """
    return DOMAIN_GRAPH_MAPPING.get(domain, DOMAIN_GRAPH_MAPPING[DomainType.GENERAL])


def detect_explicit_domain_request(query: str) -> Optional[DomainType]:
    """Detect if the user explicitly requested a specific domain.
    
    Args:
        query: The user's query
        
    Returns:
        Optional[DomainType]: The explicitly requested domain or None if not found
    """
    query_lower = query.lower()
    
    # Check for explicit domain mentions like "search in tax domain" or "using the legal database"
    domain_patterns = {
        DomainType.TAX: [r"tax\s+domain", r"tax\s+database", r"tax\s+knowledge", r"hmrc\s+database"],
        DomainType.LEGAL: [r"legal\s+domain", r"legal\s+database", r"legal\s+knowledge", r"law\s+database"],
        DomainType.FINANCE: [r"finance\s+domain", r"finance\s+database", r"financial\s+knowledge"],
        DomainType.TECHNOLOGY: [r"tech\s+domain", r"technology\s+database", r"tech\s+knowledge"],
        DomainType.HEALTHCARE: [r"health\s+domain", r"healthcare\s+database", r"medical\s+knowledge"],
        DomainType.EDUCATION: [r"education\s+domain", r"education\s+database", r"academic\s+knowledge"]
    }
    
    for domain, patterns in domain_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return domain
                
    return None


async def detect_domain(query: str, llm: Optional[LanguageModelLike] = None) -> Dict[str, Any]:
    """Detect the domain for a query using multiple methods.
    
    Args:
        query: The user's query
        llm: Optional language model for advanced domain detection
        
    Returns:
        Dict: Contains 'domain', 'confidence', and 'graph_name'
    """
    # First check for explicit domain requests
    explicit_domain = detect_explicit_domain_request(query)
    if explicit_domain:
        return {
            "domain": explicit_domain,
            "confidence": 0.9,  # High confidence for explicit requests
            "graph_name": get_graph_name_for_domain(explicit_domain),
            "method": "explicit_request"
        }
    
    # Check for URLs in the query
    urls = re.findall(r'https?://\S+|www\.\S+', query)
    if urls:
        for url in urls:
            url_domain = detect_domain_from_url(url)
            if url_domain:
                return {
                    "domain": url_domain,
                    "confidence": 0.8,  # High confidence for URL-based detection
                    "graph_name": get_graph_name_for_domain(url_domain),
                    "method": "url_analysis"
                }
    
    # Use keyword-based detection
    text_domain = detect_domain_from_text(query)
    confidence = 0.6 if text_domain != DomainType.GENERAL else 0.3
    
    # TODO: If an LLM is provided, we could use it for more sophisticated domain detection
    # This would involve prompting the LLM with the query and asking it to classify the domain
    
    return {
        "domain": text_domain,
        "confidence": confidence,
        "graph_name": get_graph_name_for_domain(text_domain),
        "method": "keyword_analysis"
    }


# Use this function to get domain information for both queries and ingestion
async def get_domain_info(text: str, llm: Optional[LanguageModelLike] = None) -> Dict[str, Any]:
    """Get comprehensive domain information for a text.
    
    Args:
        text: The text to analyze (query or content)
        llm: Optional language model for advanced domain detection
        
    Returns:
        Dict: Contains domain information including graph name
    """
    domain_info = await detect_domain(text, llm)
    
    # Add additional information that might be useful for consumers
    domain_info["keywords"] = list(DOMAIN_KEYWORDS.get(domain_info["domain"], set()))[:10]  # Just include a sample of keywords
    
    return domain_info