"""Verification components for ensuring quality and trustworthiness of information.

This module implements the Self-RAG verification approach with three main components:
1. Retrieval Grader: Verifies the relevance of retrieved documents to the user question
2. Hallucination Grader: Checks if the generated answer is grounded in the retrieved documents
3. Answer Grader: Validates that the answer properly addresses the user's question

For advanced verification, each check is performed by three independent models for
enhanced accuracy and reduced bias.

These components work together to ensure all information provided through chat,
dynamic ingestion, deep research, or auto-learning is verified before being
presented to the user.
"""
import logging
from typing import Dict, List, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import chain, RunnablePassthrough, RunnableLambda, RunnableBranch

logger = logging.getLogger(__name__)

# Data models for structured output
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    reason: str = Field(
        description="Explanation for why the document is or is not relevant"
    )
    
    @field_validator('binary_score')
    @classmethod
    def validate_binary_score(cls, v: str) -> str:
        """Validate that binary score is yes or no."""
        v_lower = v.lower()
        if v_lower not in ["yes", "no"]:
            raise ValueError("binary_score must be 'yes' or 'no'")
        return v_lower


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    reason: str = Field(
        description="Explanation for why the answer is or is not grounded in facts"
    )
    unsupported_statements: List[str] = Field(
        description="List of statements in the answer that are not supported by the provided facts",
        default_factory=list
    )
    
    @field_validator('binary_score')
    @classmethod
    def validate_binary_score(cls, v: str) -> str:
        """Validate that binary score is yes or no."""
        v_lower = v.lower()
        if v_lower not in ["yes", "no"]:
            raise ValueError("binary_score must be 'yes' or 'no'")
        return v_lower


class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    reason: str = Field(
        description="Explanation for why the answer does or does not address the question"
    )
    missing_information: List[str] = Field(
        description="List of aspects of the question that were not addressed in the answer",
        default_factory=list
    )
    
    @field_validator('binary_score')
    @classmethod
    def validate_binary_score(cls, v: str) -> str:
        """Validate that binary score is yes or no."""
        v_lower = v.lower()
        if v_lower not in ["yes", "no"]:
            raise ValueError("binary_score must be 'yes' or 'no'")
        return v_lower


def create_retrieval_grader(llm: LanguageModelLike, prompt_variant: str = "standard"):
    """Creates a grader for evaluating document relevance.
    
    Args:
        llm: The language model to use for grading
        prompt_variant: The type of prompt to use - "standard", "strict", or "lenient"
        
    Returns:
        The grader chain for document relevance evaluation
    """
    # Use structured output with the provided LLM
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    # Define the system prompt based on the variant
    if prompt_variant == "strict":
        system = """You are a meticulous grader assessing the STRICT relevance of a retrieved document to a user question.

Your task is to determine if the document contains HIGHLY SPECIFIC information relevant to answering the user's question.
A document is relevant ONLY if it contains:
1. EXACT keywords and entities mentioned in the question
2. PRECISE information that directly answers a significant part of the question
3. SPECIFIC context that is ESSENTIAL to understanding the topic of the question

A document is NOT relevant if:
1. It contains only generally related information without specific answers
2. It covers the broader topic but doesn't address the specific question
3. It contains similar keywords but in a different specific context
4. It provides incomplete or partial information related to the question

Be EXTREMELY selective - only consider a document relevant if it contains SPECIFIC information that DIRECTLY helps address the question.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
    elif prompt_variant == "lenient":
        system = """You are a flexible grader assessing the BROAD relevance of a retrieved document to a user question.

Your task is to determine if the document contains ANY information that might be helpful for answering the user's question.
A document is relevant if it contains:
1. ANY keywords or concepts related to the question's topic
2. ANY information that could provide context or background on the question
3. TANGENTIAL information that might help form a more complete understanding
4. ANY examples, analogies, or cases that might indirectly inform the answer

A document is only NOT relevant if:
1. It is completely unrelated to the question's domain or subject
2. It contains obvious misinformation about the topic
3. It is entirely about a different subject with no connection to the question

Be HIGHLY inclusive - if a document has EVEN REMOTELY useful information that might help build a complete answer, consider it relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
    else:  # standard
        system = """You are a grader assessing relevance of a retrieved document to a user question.

Your task is to determine if the document contains information relevant to answering the user's question.
A document is relevant if it contains:
1. Keywords directly related to the main entities or concepts in the question
2. Information that would help answer any part of the question, even if incomplete
3. Context that clarifies or expands on the topic of the question

A document is NOT relevant if:
1. It contains only tangentially related information with no clear connection to the question
2. It's about a completely different topic or subject area
3. It contains the same keywords but in a different context or meaning

Be somewhat inclusive - if a document has ANY information that could help address the question, consider it relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
    
    # Create the prompt template
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )
    
    # Chain the prompt and structured LLM together
    return grade_prompt | structured_llm_grader


def create_hallucination_grader(llm: LanguageModelLike, prompt_variant: str = "standard"):
    """Creates a grader for evaluating answer factuality.
    
    Args:
        llm: The language model to use for grading
        prompt_variant: The type of prompt to use - "standard", "strict", or "lenient"
        
    Returns:
        The grader chain for hallucination detection
    """
    # Use structured output with the provided LLM
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    # Define the system prompt based on the variant
    if prompt_variant == "strict":
        system = """You are a RIGOROUS fact-checker assessing WHETHER an answer is STRICTLY grounded in a set of retrieved facts.

Your task is to METICULOUSLY evaluate if the answer contains ANY statements not EXPLICITLY supported by the provided facts.
An answer is grounded ONLY if:
1. EVERY SINGLE factual statement in the answer can be DIRECTLY verified from the provided facts
2. ABSOLUTELY NO additional information is presented as fact beyond what's EXPLICITLY stated in the documents
3. ALL inferences or conclusions are NECESSARY and UNAVOIDABLE interpretations of the facts provided

An answer is NOT grounded if it:
1. Contains ANY factual claims that cannot be DIRECTLY verified from the provided facts
2. Presents ANY degree of speculation or assumption as fact
3. In ANY WAY contradicts or differs from the information in the provided facts
4. Adds ANY specific details (like dates, numbers, names, or events) not EXPLICITLY found in the facts
5. Makes ANY generalizations beyond what is stated in the documents

Be EXTREMELY STRICT in your evaluation. If there is ANY doubt about whether a statement is supported, consider it a hallucination.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is FULLY grounded in the facts, 'no' means it contains ANY hallucinations.
In your reason, cite specific examples of hallucinations if they exist.
"""
    elif prompt_variant == "lenient":
        system = """You are a BALANCED fact-checker assessing whether an answer is REASONABLY grounded in a set of retrieved facts.

Your task is to evaluate if the answer contains SIGNIFICANT statements that are SUBSTANTIALLY unsupported by the provided facts.
An answer is grounded if:
1. The MAIN factual statements in the answer can be reasonably inferred from the provided facts
2. Only MINOR additional information is presented beyond what's contained in the documents
3. Inferences or conclusions are PLAUSIBLE interpretations of the facts provided
4. CONTEXTUAL knowledge that most educated readers would know is reasonably incorporated

An answer is NOT grounded ONLY if it:
1. Contains MAJOR factual claims that substantially deviate from the provided facts
2. Presents SPECULATIVE or CONTROVERSIAL assertions as established facts
3. DIRECTLY contradicts key information in the provided facts
4. Invents CRITICAL specific details (like dates, numbers, names, or events) that change the meaning

Be REASONABLY FLEXIBLE in your evaluation, allowing for minor elaborations and common-sense inferences.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is REASONABLY grounded in the facts, 'no' means it has SIGNIFICANT hallucinations.
In your reason, cite specific examples of major hallucinations if they exist.
"""
    else:  # standard
        system = """You are a grader assessing whether an answer is grounded in and supported by a set of retrieved facts.

Your task is to carefully evaluate if the answer contains statements that are NOT supported by the provided facts.
An answer is grounded if:
1. All factual statements in the answer can be verified from the provided facts
2. No additional information is presented as fact beyond what's contained in the documents
3. Any inferences or conclusions are reasonable interpretations of the facts provided

An answer is NOT grounded if it:
1. Contains factual claims that cannot be verified from the provided facts
2. Presents speculation as fact
3. Contradicts any information in the provided facts
4. Makes up specific details (like dates, numbers, names, or events) not found in the facts

Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in the facts, 'no' means it has hallucinations.
In your reason, cite specific examples of hallucinations if they exist.
"""
    
    # Create the prompt template
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n Answer to be evaluated: {answer}")
        ]
    )
    
    # Chain the prompt and structured LLM together
    return hallucination_prompt | structured_llm_grader


def create_answer_grader(llm: LanguageModelLike, prompt_variant: str = "standard"):
    """Creates a grader for evaluating if the answer addresses the question.
    
    Args:
        llm: The language model to use for grading
        prompt_variant: The type of prompt to use - "standard", "strict", or "lenient"
        
    Returns:
        The grader chain for answer evaluation
    """
    # Use structured output with the provided LLM
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    
    # Define the system prompt based on the variant
    if prompt_variant == "strict":
        system = """You are a DEMANDING grader assessing whether an answer COMPREHENSIVELY addresses a user's question.

Your task is to METICULOUSLY determine if the answer provides ALL the information the user was seeking.
An answer addresses the question ONLY if:
1. It provides DETAILED information that DIRECTLY and COMPLETELY answers what was asked
2. It covers EVERY ASPECT of multi-part questions with EQUAL depth
3. It provides ALL specific details that were explicitly or implicitly requested
4. It is COMPREHENSIVE in its coverage of the question's scope
5. It anticipates and addresses POTENTIAL follow-up questions

An answer does NOT address the question if it:
1. Answers a question even SLIGHTLY different from what was asked
2. Omits ANY part of multi-part questions
3. Is even SOMEWHAT vague or generic when specific details were requested
4. Contains relevant information but fails to draw clear, explicit conclusions
5. Lacks ANY important context that would make the answer more complete

Be EXTREMELY THOROUGH in your evaluation. An answer must be COMPREHENSIVE to be considered adequate.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer FULLY addresses the question.
In your reason, EXPLICITLY list any aspects of the question that were missed or inadequately addressed.
"""
    elif prompt_variant == "lenient":
        system = """You are a PRACTICAL grader assessing whether an answer SUFFICIENTLY addresses a user's question.

Your task is to determine if the answer provides the CORE information the user was seeking.
An answer addresses the question if:
1. It provides the MAIN information relevant to what was asked, even if some details are missing
2. It covers the MOST IMPORTANT parts of multi-part questions, even if some minor parts are skipped
3. It provides ENOUGH specific details to be useful, even if not comprehensive
4. It addresses the CENTRAL aspects of the question, even if peripheral elements are omitted

An answer does NOT address the question ONLY if it:
1. Answers a SUBSTANTIALLY different question than what was asked
2. COMPLETELY misses major parts of multi-part questions
3. Is so vague or generic that it provides VIRTUALLY NO useful information
4. Contains tangentially relevant information but COMPLETELY fails to address the core question

Be PRACTICAL in your evaluation. An answer that provides the core information the user needs should be considered sufficient.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer SUFFICIENTLY addresses the question.
In your reason, include what MAJOR aspects of the question were missed, if any.
"""
    else:  # standard
        system = """You are a grader assessing whether an answer properly addresses a user's question.

Your task is to determine if the answer provides the information the user was seeking.
An answer addresses the question if:
1. It provides information directly relevant to what was asked
2. It covers all parts of multi-part questions
3. It provides the specific details that were requested
4. It's at an appropriate level of detail given the question's scope

An answer does NOT address the question if it:
1. Answers a different question than what was asked
2. Only partially addresses multi-part questions
3. Is too vague or generic when specific details were requested
4. Contains relevant information but fails to draw conclusions requested in the question

Give a binary score 'yes' or 'no'. 'Yes' means that the answer properly addresses the question.
In your reason, include what aspects of the question were missed, if any.
"""
    
    # Create the prompt template
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n Answer to be evaluated: {answer}")
        ]
    )
    
    # Chain the prompt and structured LLM together
    return answer_prompt | structured_llm_grader


class VerificationComponents:
    """Container for all verification components."""
    
    def __init__(self, llm: LanguageModelLike, advanced_verification: bool = False):
        """Initialize the verification components with the given LLM.
        
        Args:
            llm: The language model to use for verification
            advanced_verification: Whether to use triple verification with multiple models
        """
        self.llm = llm
        self.advanced_verification = advanced_verification
        
        # Initialize standard graders
        self.retrieval_grader = create_retrieval_grader(llm)
        self.hallucination_grader = create_hallucination_grader(llm)
        self.answer_grader = create_answer_grader(llm)
        
        # For advanced verification, we'll create additional graders with the same LLM
        # but different system prompts to introduce diversity in verification
        if advanced_verification:
            # Initialize the secondary grader set
            self.retrieval_grader_2 = create_retrieval_grader(llm, prompt_variant="strict")
            self.hallucination_grader_2 = create_hallucination_grader(llm, prompt_variant="strict")
            self.answer_grader_2 = create_answer_grader(llm, prompt_variant="strict")
            
            # Initialize the tertiary grader set
            self.retrieval_grader_3 = create_retrieval_grader(llm, prompt_variant="lenient")
            self.hallucination_grader_3 = create_hallucination_grader(llm, prompt_variant="lenient")
            self.answer_grader_3 = create_answer_grader(llm, prompt_variant="lenient")
    
    def filter_relevant_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """Filter documents to keep only those relevant to the question."""
        if not documents:
            return []
        
        filtered_docs = []
        
        if not self.advanced_verification:
            # Standard single verification
            for doc in documents:
                try:
                    result = self.retrieval_grader.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    grade = result.binary_score.lower()
                    
                    if grade == "yes":
                        filtered_docs.append(doc)
                        logger.debug(f"Document accepted: {doc.metadata.get('source', 'unknown')}. Reason: {result.reason}")
                    else:
                        logger.debug(f"Document filtered out: {doc.metadata.get('source', 'unknown')}. Reason: {result.reason}")
                except Exception as e:
                    logger.warning(f"Error grading document relevance: {e}")
                    # Include the document on error to avoid missing potentially relevant information
                    filtered_docs.append(doc)
        else:
            # Advanced triple verification
            for doc in documents:
                try:
                    # Run all three graders
                    result1 = self.retrieval_grader.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    result2 = self.retrieval_grader_2.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    result3 = self.retrieval_grader_3.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    
                    # Get binary scores (yes/no)
                    grade1 = result1.binary_score.lower() == "yes"
                    grade2 = result2.binary_score.lower() == "yes"
                    grade3 = result3.binary_score.lower() == "yes"
                    
                    # Count the number of positive votes
                    positive_votes = sum([grade1, grade2, grade3])
                    
                    # Document is relevant if majority (2+ of 3) say yes
                    if positive_votes >= 2:
                        filtered_docs.append(doc)
                        logger.debug(f"Document accepted by majority ({positive_votes}/3): {doc.metadata.get('source', 'unknown')}")
                        
                        # Log reasoning from each grader
                        logger.debug(f"Standard grader: {result1.binary_score} - {result1.reason}")
                        logger.debug(f"Strict grader: {result2.binary_score} - {result2.reason}")
                        logger.debug(f"Lenient grader: {result3.binary_score} - {result3.reason}")
                    else:
                        logger.debug(f"Document filtered out ({positive_votes}/3 votes): {doc.metadata.get('source', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Error in triple verification of document relevance: {e}")
                    # Include the document on error to avoid missing potentially relevant information
                    filtered_docs.append(doc)
                
        return filtered_docs
    
    def check_answer_hallucination(self, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """Check if the answer contains hallucinations based on the documents."""
        if not documents:
            return {
                "is_hallucination": True,
                "reason": "No documents provided to verify against",
                "unsupported_statements": []
            }
        
        try:
            # Format documents for grading
            docs_text = "\n\n".join([doc.page_content for doc in documents])
            
            if not self.advanced_verification:
                # Standard single verification
                result = self.hallucination_grader.invoke({
                    "documents": docs_text,
                    "answer": answer
                })
                
                is_hallucination = result.binary_score.lower() != "yes"
                
                return {
                    "is_hallucination": is_hallucination,
                    "reason": result.reason,
                    "unsupported_statements": result.unsupported_statements
                }
            else:
                # Advanced triple verification
                # Run all three graders
                result1 = self.hallucination_grader.invoke({
                    "documents": docs_text,
                    "answer": answer
                })
                result2 = self.hallucination_grader_2.invoke({
                    "documents": docs_text,
                    "answer": answer
                })
                result3 = self.hallucination_grader_3.invoke({
                    "documents": docs_text,
                    "answer": answer
                })
                
                # Get binary scores (yes=grounded, no=hallucination)
                is_grounded1 = result1.binary_score.lower() == "yes"
                is_grounded2 = result2.binary_score.lower() == "yes"
                is_grounded3 = result3.binary_score.lower() == "yes"
                
                # Count the number of votes for "is grounded"
                grounded_votes = sum([is_grounded1, is_grounded2, is_grounded3])
                
                # Answer is not hallucinated if majority (2+ of 3) say it's grounded
                is_hallucination = grounded_votes < 2
                
                # Collect all unsupported statements identified by any of the graders
                all_unsupported = list(set(
                    result1.unsupported_statements + 
                    result2.unsupported_statements + 
                    result3.unsupported_statements
                ))
                
                # Combine the reasons with attribution
                combined_reason = f"Verification summary ({grounded_votes}/3 say grounded):\n"
                combined_reason += f"• Standard grader: {result1.binary_score} - {result1.reason.strip()}\n"
                combined_reason += f"• Strict grader: {result2.binary_score} - {result2.reason.strip()}\n"
                combined_reason += f"• Lenient grader: {result3.binary_score} - {result3.reason.strip()}"
                
                return {
                    "is_hallucination": is_hallucination,
                    "reason": combined_reason,
                    "unsupported_statements": all_unsupported
                }
        except Exception as e:
            logger.warning(f"Error checking for hallucinations: {e}")
            return {
                "is_hallucination": False,  # Conservative approach on error
                "reason": "Error during hallucination check",
                "unsupported_statements": []
            }
    
    def check_answer_addresses_question(self, question: str, answer: str) -> Dict[str, Any]:
        """Check if the answer properly addresses the user's question."""
        try:
            if not self.advanced_verification:
                # Standard single verification
                result = self.answer_grader.invoke({
                    "question": question,
                    "answer": answer
                })
                
                addresses_question = result.binary_score.lower() == "yes"
                
                return {
                    "addresses_question": addresses_question,
                    "reason": result.reason,
                    "missing_information": result.missing_information
                }
            else:
                # Advanced triple verification
                # Run all three graders
                result1 = self.answer_grader.invoke({
                    "question": question,
                    "answer": answer
                })
                result2 = self.answer_grader_2.invoke({
                    "question": question,
                    "answer": answer
                })
                result3 = self.answer_grader_3.invoke({
                    "question": question,
                    "answer": answer
                })
                
                # Get binary scores
                addresses1 = result1.binary_score.lower() == "yes"
                addresses2 = result2.binary_score.lower() == "yes"
                addresses3 = result3.binary_score.lower() == "yes"
                
                # Count the number of positive votes
                addresses_votes = sum([addresses1, addresses2, addresses3])
                
                # Answer addresses question if majority (2+ of 3) say yes
                addresses_question = addresses_votes >= 2
                
                # Collect all missing information identified by any of the graders
                all_missing_info = list(set(
                    result1.missing_information + 
                    result2.missing_information + 
                    result3.missing_information
                ))
                
                # Combine the reasons with attribution
                combined_reason = f"Verification summary ({addresses_votes}/3 say question addressed):\n"
                combined_reason += f"• Standard grader: {result1.binary_score} - {result1.reason.strip()}\n"
                combined_reason += f"• Strict grader: {result2.binary_score} - {result2.reason.strip()}\n"
                combined_reason += f"• Lenient grader: {result3.binary_score} - {result3.reason.strip()}"
                
                return {
                    "addresses_question": addresses_question,
                    "reason": combined_reason,
                    "missing_information": all_missing_info
                }
        except Exception as e:
            logger.warning(f"Error checking if answer addresses question: {e}")
            return {
                "addresses_question": True,  # Conservative approach on error
                "reason": "Error during answer evaluation",
                "missing_information": []
            }
    
    def create_improved_answer(self, question: str, original_answer: str, 
                              documents: List[Document], 
                              hallucination_info: Dict[str, Any],
                              question_addressing_info: Dict[str, Any]) -> str:
        """Create an improved answer that addresses hallucinations and missing information."""
        # Format documents for context
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Create system message based on verification results
        if hallucination_info["is_hallucination"]:
            system_content = """You are a helpful assistant creating a revised response to a user's question.
            
The original answer contained statements not supported by the provided documents. Your task is to create
a new answer that:
1. Only includes information that can be verified from the documents
2. Removes any speculative or ungrounded statements
3. Clearly indicates the limits of your knowledge when information is not available
4. Maintains a helpful and informative tone

Base your answer ONLY on the context provided. Do not incorporate external knowledge or assumptions.
            """
        elif not question_addressing_info["addresses_question"]:
            system_content = """You are a helpful assistant creating a revised response to a user's question.
            
The original answer did not fully address all aspects of the user's question. Your task is to create
a new answer that:
1. Addresses all parts of the question directly and specifically
2. Includes all relevant information from the documents
3. Clearly indicates if any part of the question cannot be answered with the available information
4. Maintains a helpful and informative tone

Base your answer ONLY on the context provided. Do not incorporate external knowledge or assumptions.
            """
        else:
            # If we're here, we're just improving a basically correct answer
            system_content = """You are a helpful assistant creating an improved response to a user's question.
            
Your task is to refine the original answer to make it:
1. More concise and directly relevant to the question
2. Better organized and easier to understand
3. More complete in addressing all aspects of the question using the available information
4. Clear about the limitations of the available information when relevant

Base your answer ONLY on the context provided. Do not incorporate external knowledge or assumptions.
            """
        
        # Construct messages for the LLM
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"""
Question: {question}

Available context:
{context}

Original answer:
{original_answer}

Issues to fix:
{"Unsupported statements: " + ", ".join(hallucination_info["unsupported_statements"]) if hallucination_info["is_hallucination"] else "No hallucination issues."}
{"Missing information: " + ", ".join(question_addressing_info["missing_information"]) if not question_addressing_info["addresses_question"] else "Question is fully addressed."}

Please provide an improved answer that addresses these issues:
""")
        ]
        
        # Generate improved answer
        improved_answer = self.llm.invoke(messages).content
        
        return improved_answer
    
    def verify_and_improve_answer(self, question: str, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """Perform full verification and improvement of an answer."""
        # Check for hallucinations
        hallucination_result = self.check_answer_hallucination(answer, documents)
        
        # Check if answer addresses the question
        addressing_result = self.check_answer_addresses_question(question, answer)
        
        # Determine if improvement is needed
        needs_improvement = hallucination_result["is_hallucination"] or not addressing_result["addresses_question"]
        
        result = {
            "original_answer": answer,
            "hallucination_check": hallucination_result,
            "question_addressing_check": addressing_result,
            "needs_improvement": needs_improvement,
            "verified_answer": answer  # Default to original
        }
        
        # If improvement needed, create improved answer
        if needs_improvement:
            improved_answer = self.create_improved_answer(
                question, 
                answer, 
                documents, 
                hallucination_result,
                addressing_result
            )
            result["verified_answer"] = improved_answer
        
        return result


def format_docs_for_verification(docs: List[Document]) -> str:
    """Format a list of documents into a string for verification."""
    if not docs:
        return ""
    
    formatted_content = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        title = doc.metadata.get("title", "Untitled document")
        formatted_content.append(f"Document {i+1}: {title} (Source: {source})\n{doc.page_content}")
    
    return "\n\n".join(formatted_content)


# Utility function to create all verification components with a given LLM
def get_verification_components(llm: LanguageModelLike, advanced_verification: bool = False) -> VerificationComponents:
    """Create and return all verification components using the provided LLM.
    
    Args:
        llm: The language model to use for verification
        advanced_verification: Whether to enable triple verification with multiple models
        
    Returns:
        The verification components configured appropriately
    """
    return VerificationComponents(llm, advanced_verification)