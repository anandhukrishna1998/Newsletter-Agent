from dotenv import load_dotenv
from phi.agent import Agent
import phi.api
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.resend_tools import ResendTools  # ResendTools properly imported
import os
import phi
from phi.tools.googlesearch import GoogleSearch
from phi.model.google import Gemini
from datetime import datetime  

# Load environment variables from .env file
load_dotenv()

# Set the API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
phi.api = os.getenv("PHI_API_KEY")

current_date = datetime.now().strftime("%Y-%m-%d")

# Instantiate ResendTools
resend_tool = ResendTools(
    api_key=os.getenv("RESEND_API_KEY"),
    from_email=os.getenv("from_email")
)

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="You are a web search analyst responsible for finding the latest AI news.",
    # model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    model=Gemini(id="gemini-1.5-flash"),
    tools=[ DuckDuckGo()],
    instructions=[
        f"Search for the top 6 latest **content pieces** (news, articles, blog posts, reports, papers, or discussions) in **AI, LLM, GenAI, Multimodal AI, and AI Governance** that happened in the **last 24 hours globally**, based on the current date: {current_date}.",
        "Ensure the content is highly relevant to these domains and comes from reputable sources with real-time updates.",
        "The content must be **from the last 24 hours** only. Use the current date and time to filter content published within the past 24 hours.",
        "Include a variety of formats, such as articles, research papers, news pieces, blog posts, or detailed reports.",
        "For each piece of content, include the title of the article/report in an `<h2>` tag.",
        "Include a brief **4-5 line summary** or description of the content inside a `<p>` tag.",
        "Add a clickable `<a>` tag linking to the source for each content piece.",
        "Ensure that all source links are valid and lead to the actual content. Check for broken links or 404 errors before including the link.",
        "If any source link results in a 404 error or is broken, exclude it from the newsletter and replace it with a note stating 'Link not found' or use an alternative valid source.",
        "Return the entire set of results in a **structured HTML format** that is valid and ready to be rendered by a browser.",
        "Ensure all HTML tags are properly closed and indented for readability.",
        "Avoid plain text or markdown formatting. Output only raw HTML content.",
        "Use diverse content sources, including trusted AI blogs, industry reports, research papers, news websites, and academic papers.",
        "Examples of sources could include: TechCrunch AI, Google AI Blog, Arxiv papers, ResearchGate, Wired, AI Weekly, VentureBeat, and more.",
        "Make sure to include **all types of AI-related content**, not just news articles but also reports, blogs, discussions, and papers from the past 24 hours.",
        "Ensure that **OpenAI**, **Anthropic**, **Meta**, **HuggingFace**, **Google DeepMind**, **Stanford AI**, **MIT AI**, **Microsoft AI**, and **AI organizations** such as **AI Weekly**, **VentureBeat**, **Wired AI**, and **arXiv** are part of the search pool.",
        "Consider incorporating other **leading AI labs and researchers** such as those at **DeepMind**, **Meta AI Research**, **Stanford AI**, and **academic papers**.",
        "Include reports from **academic institutions** like **MIT AI**, **Stanford's AI Lab**, **Carnegie Mellon AI**, or **University of California AI Research**.",
        "Incorporate relevant articles, research, or discussions from AI conferences and summits that occurred within the last 24 hours, such as **NeurIPS**, **ICLR**, **AAAI**, **CVPR**, **ICML**, etc."
    ],
    show_tools_calls=True,
    markdown=True,
)
# Email Send Agent
email_send_agent = Agent(
    name="Email Send Agent",
    role="You are responsible for sending emails containing AI newsletters.",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[resend_tool],
    instructions=[
        "Ensure the email includes all the news items formatted as HTML.",
        "Always include sources.",
        "Use the Resend API to send the email."
    ],
    show_tool_calls=True,
    markdown=True,
)


# Execution
def fetch_and_send_newsletter():
    # Step 1: Fetch latest news
    response = web_search_agent.run("Fetch the latest AI news and format it as an HTML newsletter.")
    print(response.content)
    html_content = response.content.strip().replace('```html', '').replace('', '').strip()
    print(html_content)

    # Step 2: Send the email with the newsletter
    email_response = resend_tool.send_email(
        to_email=os.getenv("to_email"),
        subject="AI Newsletter - " + current_date,
        body=html_content
    )
    return email_response


# Run the process
print(fetch_and_send_newsletter())
