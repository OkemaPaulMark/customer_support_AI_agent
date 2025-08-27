import os
import re
import uuid
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "ai_agent")


# Database Connection
def get_db_connection():
    """Establish and return a database connection."""
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD or "",
            database=MYSQL_DB,
            cursorclass=pymysql.cursors.DictCursor,
        )
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {e}") from e


# Ticket Table Initialization
def init_ticket_db():
    """Initialize the tickets table if it doesn't exist."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticket_id VARCHAR(20) UNIQUE,
                    user_name VARCHAR(255),
                    issue TEXT,
                    response TEXT,
                    status VARCHAR(50) DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
        print("Ticket table initialized successfully.")
        return True
    except ConnectionError as e:
        print(f"Failed to connect to database for initialization: {e}")
        return False
    except Exception as e:
        print(f"Error initializing ticket table: {e}")
        return False
    finally:
        if conn:
            conn.close()


# Support Ticket Functions
def create_support_ticket(user_name: str, issue: str):
    """Insert a new support ticket into the database."""
    conn = None
    try:
        conn = get_db_connection()

        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO tickets (ticket_id, user_name, issue, status) 
                VALUES (%s, %s, %s, %s)
                """,
                (ticket_id, user_name, issue, "open"),
            )
            conn.commit()
            return ticket_id
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return None
    except Exception as e:
        print(f"Error creating support ticket: {e}")
        return None
    finally:
        if conn:
            conn.close()


def check_ticket_status(ticket_id: str):
    """Check the status of an existing support ticket."""
    conn = None
    try:
        conn = get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT status, created_at FROM tickets WHERE ticket_id = %s",
                (ticket_id,),
            )
            result = cursor.fetchone()

            if result:
                status = result["status"]
                created_at = result["created_at"]
                return f"Ticket {ticket_id} is {status} (created: {created_at})"
            else:
                return f"Could not find ticket {ticket_id}."
    except ConnectionError as e:
        return f"Database connection failed: {e}"
    except Exception as e:
        return f"Error checking ticket status: {str(e)}"
    finally:
        if conn:
            conn.close()


def update_ticket_response(ticket_id: str, response: str, status: str = "closed"):
    """Human support can respond to a ticket and update its status."""
    conn = None
    try:
        conn = get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE tickets 
                SET response = %s, status = %s 
                WHERE ticket_id = %s
                """,
                (response, status, ticket_id),
            )
            conn.commit()
            return True
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return False
    except Exception as e:
        print(f"Error updating ticket response: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_ticket_by_id(ticket_id: str):
    """Fetch a ticket including the human response."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT ticket_id, user_name, issue, response, status, created_at FROM tickets WHERE ticket_id = %s",
                (ticket_id,),
            )
            return cursor.fetchone()
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return None
    except Exception as e:
        print(f"Error fetching ticket: {e}")
        return None
    finally:
        if conn:
            conn.close()


def query_ticket_answer(question: str):
    """
    Check if a past ticket exists with the same issue and has a response.
    Returns the response if found, otherwise None.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT response FROM tickets WHERE issue = %s AND response IS NOT NULL ORDER BY created_at DESC LIMIT 1",
                (question,),
            )
            result = cursor.fetchone()
            if result:
                return result["response"]
            return None
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return None
    except Exception as e:
        print(f"Error querying past ticket: {e}")
        return None
    finally:
        if conn:
            conn.close()


# Team & FAQ Query Functions
def query_team_info(question: str):
    """Search 'teams' table for team member information."""
    name = extract_name_from_question(question)
    if not name:
        return None

    conn = None
    try:
        conn = get_db_connection()

        with conn.cursor() as cursor:
            # Exact match
            cursor.execute(
                "SELECT name, bio FROM teams WHERE LOWER(name) = %s LIMIT 1",
                (name.lower(),),
            )
            result = cursor.fetchone()
            if result:
                return format_team_response(result)

            # Partial match
            cursor.execute(
                "SELECT name, bio FROM teams WHERE LOWER(name) LIKE %s LIMIT 1",
                (f"%{name.lower()}%",),
            )
            result = cursor.fetchone()
            if result:
                return format_team_response(result)

            return None
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return None
    except Exception as e:
        print(f"Error querying team info: {e}")
        return None
    finally:
        if conn:
            conn.close()


def query_faq(question: str, llm=None):
    """Search FAQ table using improved semantic matching with optional LLM assistance."""
    conn = None
    try:
        conn = get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'faq'")
            if not cursor.fetchone():
                return None

            keywords = extract_keywords(question)
            if keywords:
                like_conditions = []
                params = []

                for keyword in keywords:
                    like_conditions.append("LOWER(question) LIKE %s")
                    like_conditions.append("LOWER(answer) LIKE %s")
                    params.extend([f"%{keyword}%", f"%{keyword}%"])

                if like_conditions:
                    sql = f"""
                        SELECT question, answer, 
                               (CASE 
                                WHEN LOWER(question) LIKE %s THEN 1
                                WHEN LOWER(answer) LIKE %s THEN 2
                                ELSE 3
                                END) as relevance
                        FROM faq 
                        WHERE {' OR '.join(like_conditions)}
                        ORDER BY relevance, LENGTH(question)
                        LIMIT 3
                    """
                    params.extend([f"%{question.lower()}%", f"%{question.lower()}%"])
                    cursor.execute(sql, params)
                    results = cursor.fetchall()

                    if results:
                        if llm and len(results) > 1:
                            return query_faq_with_llm(question, results, llm)
                        return results[0]["answer"]

            cursor.execute(
                "SELECT question, answer FROM faq WHERE LOWER(question) LIKE %s OR LOWER(answer) LIKE %s ORDER BY LENGTH(question) LIMIT 1",
                (f"%{question.lower()}%", f"%{question.lower()}%"),
            )
            result = cursor.fetchone()
            if result:
                return result["answer"]

            return None
    except ConnectionError as e:
        print(f"Database connection failed: {e}")
        return None
    except Exception as e:
        print(f"Error querying FAQ: {e}")
        return None
    finally:
        if conn:
            conn.close()


def query_faq_with_llm(question: str, faq_results: list, llm):
    try:
        options_text = "\n".join(
            [
                f"{i+1}. Q: {faq['question']} - A: {faq['answer'][:100]}..."
                for i, faq in enumerate(faq_results)
            ]
        )
        prompt = f"""
        User Question: "{question}"
        Which of these FAQ answers best matches the user's question? 
        Return ONLY the number (1-{len(faq_results)}) of the best match:

        {options_text}
        Respond with only the number:
        """
        response = llm.invoke(prompt)
        match_index = int(response.content.strip())
        if 1 <= match_index <= len(faq_results):
            return faq_results[match_index - 1]["answer"]
    except Exception:
        pass
    return faq_results[0]["answer"]


# Utility Functions
def extract_keywords(question: str):
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "why",
        "how",
        "your",
        "you",
        "me",
        "my",
        "our",
        "we",
        "us",
        "i",
        "he",
        "she",
        "it",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "to",
        "for",
        "with",
        "by",
        "at",
        "on",
        "in",
        "of",
        "about",
        "as",
        "if",
        "then",
        "than",
        "so",
        "because",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    return list(set(word for word in words if word not in stop_words))


def extract_name_from_question(question: str):
    patterns = [
        r"(?:who is|tell me about|what does|who's|what is)\s+([a-zA-Z\s]+)(?:\'s)?",
        r"([a-zA-Z\s]+)(?:\'s)?(?:\s+profile|info|bio|other name|role)?$",
    ]
    question_lower = question.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            name = match.group(1).strip()
            if len(name) > 2:
                return name
    words = question_lower.split()
    capitalized_words = [word for word in words if word[0].isupper()]
    if capitalized_words:
        return " ".join(capitalized_words)
    return None


def format_team_response(result: dict):
    return f"{result['name']}: {result['bio']}"


def database_query_tool(question: str, llm=None):
    """
    Main database query tool that tries multiple tables.
    Returns formatted response if found, or None if no info.
    """
    # For general questions, try FAQ first (with LLM assistance)
    if is_general_question(question):
        faq_result = query_faq(question, llm)
        if faq_result:
            return {"response": faq_result, "found": True}

    # Then try team info (for people questions)
    team_result = query_team_info(question)
    if team_result:
        return {"response": team_result, "found": True}

    # If FAQ wasn't tried first, try it now
    if not is_general_question(question):
        faq_result = query_faq(question, llm)
        if faq_result:
            return {"response": faq_result, "found": True}

    # Nothing found
    return {"response": "No information found.", "found": False}


def is_general_question(question: str):
    general_keywords = [
        "what",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "would",
        "should",
        "hours",
        "time",
        "open",
        "close",
        "location",
        "address",
        "contact",
        "phone",
        "email",
        "price",
        "cost",
        "service",
        "support",
        "help",
        "business",
        "work",
        "operating",
        "available",
        "hour",
        "schedule",
        "timing",
    ]
    return any(keyword in question.lower() for keyword in general_keywords)


def check_database_connection():
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return True
        return False
    except ConnectionError:
        return False
