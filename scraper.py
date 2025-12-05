from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

BASE_URL = "https://huggingface.co"


async def scrape_paper_details(crawler, paper_url):
    """Scrape individual paper page for detailed information"""
    try:
        result = await crawler.arun(url=paper_url)
        soup = BeautifulSoup(result.html, "html.parser")

        # Title
        title = soup.select_one("h1")
        title = title.get_text(strip=True) if title else None

        # Abstract
        abstract_block = soup.select_one("div.prose")
        abstract = abstract_block.get_text(" ", strip=True) if abstract_block else None

        # arXiv and PDF Links
        arxiv_page_url = None
        pdf_url = None
        
        for a in soup.select("a[href*='arxiv.org']"):
            href = a.get("href", "")
            if "/abs/" in href:
                arxiv_page_url = href
            elif "/pdf/" in href or href.endswith(".pdf"):
                pdf_url = href

        # GitHub Repos (deduplicated)
        github_links = list(set([
            a.get("href") for a in soup.select("a[href*='github.com']")
            if a.get("href")
        ]))

        # Metadata
        metadata = {}
        for dl in soup.select("dl"):
            dt = dl.select_one("dt")
            dd = dl.select_one("dd")
            if dt and dd:
                key = dt.get_text(strip=True)
                value = dd.get_text(strip=True)
                metadata[key] = value

        return {
            "title": title,
            "abstract": abstract,
            "arxiv_page_url": arxiv_page_url,
            "pdf_url": pdf_url,
            "github_links": github_links,
            "metadata": metadata,
            "page_url": paper_url,
            "scraped_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error scraping {paper_url}: {e}")
        return {
            "error": str(e),
            "page_url": paper_url
        }


async def scrape_hf_papers():
    """Scrape all papers from HuggingFace daily papers page"""
    async with AsyncWebCrawler() as crawler:
        print("Fetching papers listing...")
        listing = await crawler.arun(url=f"{BASE_URL}/papers")
        soup = BeautifulSoup(listing.html, "html.parser")

        cards = soup.select("div.from-gray-50-to-white")
        total_found = len(cards)
        print(f"Found {total_found} papers on listing page")
        print(f"Scraping ALL {total_found} papers...")

        all_papers = []

        for idx, card in enumerate(cards, 1):
            print(f"\n[{idx}/{len(cards)}] Processing paper...")
            
            # Title + link
            title_tag = card.select_one("h3 a")
            title = title_tag.get_text(strip=True) if title_tag else None
            paper_url = BASE_URL + title_tag.get("href") if title_tag else None
            
            if not paper_url:
                print("  ⚠️  No URL found, skipping")
                continue

            print(f"  Title: {title}")

            # Authors
            authors = [a.get("title") for a in card.select("ul li[title]") if a.get("title")]

            # Star count
            star_tag = card.select_one("a.flex span")
            stars = star_tag.get_text(strip=True) if star_tag else "0"

            # Deep scraping
            print(f"  Fetching details from {paper_url}...")
            details = await scrape_paper_details(crawler, paper_url)

            # Merge
            paper_data = {
                "title": title,
                "paper_url": paper_url,
                "authors": authors,
                "stars": stars,
                "details": details
            }
            
            all_papers.append(paper_data)
            
            # Be respectful - add delay
            if idx < len(cards):
                await asyncio.sleep(1.0)

        return all_papers


def save_to_json(data):
    """Save scraped data with daily, weekly, and monthly archives"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    week_str = today.strftime("%Y-W%W")  # Week number
    month_str = today.strftime("%Y-%m")
    
    # Create archive directories
    Path("data/daily").mkdir(parents=True, exist_ok=True)
    Path("data/weekly").mkdir(parents=True, exist_ok=True)
    Path("data/monthly").mkdir(parents=True, exist_ok=True)
    
    # Add scrape date to each paper
    for paper in data:
        paper["scraped_date"] = today_str
    
    # Save daily snapshot
    daily_file = f"data/daily/{today_str}.json"
    with open(daily_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved daily snapshot: {daily_file}")
    
    # Save/update weekly archive
    weekly_file = f"data/weekly/{week_str}.json"
    weekly_data = []
    if Path(weekly_file).exists():
        with open(weekly_file, "r", encoding="utf-8") as f:
            weekly_data = json.load(f)
    weekly_data.extend(data)
    with open(weekly_file, "w", encoding="utf-8") as f:
        json.dump(weekly_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated weekly archive: {weekly_file} (Total: {len(weekly_data)} papers)")
    
    # Save/update monthly archive
    monthly_file = f"data/monthly/{month_str}.json"
    monthly_data = []
    if Path(monthly_file).exists():
        with open(monthly_file, "r", encoding="utf-8") as f:
            monthly_data = json.load(f)
    monthly_data.extend(data)
    with open(monthly_file, "w", encoding="utf-8") as f:
        json.dump(monthly_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated monthly archive: {monthly_file} (Total: {len(monthly_data)} papers)")
    
    # Save latest for easy access
    with open("data/latest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved latest papers: data/latest.json")


def load_recent_papers(days=7):
    """Load papers from the last N days"""
    papers = []
    today = datetime.now()
    
    for i in range(days):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_file = f"data/daily/{date}.json"
        
        if Path(daily_file).exists():
            with open(daily_file, "r", encoding="utf-8") as f:
                daily_papers = json.load(f)
                papers.extend(daily_papers)
    
    return papers


def generate_readme(papers):
    """Generate an enhanced README.md with promotional content"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    today_display = today.strftime("%B %d, %Y")
    week_str = today.strftime("%Y-W%W")
    month_str = today.strftime("%Y-%m")
    
    # Load week and month totals
    weekly_file = f"data/weekly/{week_str}.json"
    monthly_file = f"data/monthly/{month_str}.json"
    
    weekly_count = 0
    if Path(weekly_file).exists():
        with open(weekly_file, "r", encoding="utf-8") as f:
            weekly_count = len(json.load(f))
    
    monthly_count = 0
    if Path(monthly_file).exists():
        with open(monthly_file, "r", encoding="utf-8") as f:
            monthly_count = len(json.load(f))
    
    # Count total papers in all archives
    total_papers = 0
    if Path("data/daily").exists():
        for daily_file in Path("data/daily").glob("*.json"):
            with open(daily_file, "r", encoding="utf-8") as f:
                total_papers += len(json.load(f))
    
    readme = f"""<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-{len(papers)}-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-{total_papers}+-orange?style=for-the-badge&logo=academia)](data/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/AtharvaDomale/Daily-HuggingFace-AI-Papers?style=social)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/stargazers)

**Automatically updated every day at 00:00 UTC** ⏰

[📊 View Data](data/) | [🔍 Latest Papers](data/latest.json) | [📅 Archives](#-historical-archives) | [⭐ Star This Repo](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers)

</div>

---

## 🎯 Why This Repo?

- ✅ **Saves 30+ minutes** of daily paper hunting
- ✅ **Organized archives** - daily, weekly, and monthly snapshots
- ✅ **Direct links** to arXiv, PDFs, and GitHub repositories
- ✅ **Machine-readable JSON** format for easy integration
- ✅ **Zero maintenance** - fully automated via GitHub Actions
- ✅ **Historical data** - track AI research trends over time

---

## 🚀 Who Is This For?

<table>
<tr>
<td align="center">🔬<br/><b>Researchers</b><br/>Stay current with latest developments</td>
<td align="center">💼<br/><b>ML Engineers</b><br/>Discover SOTA techniques</td>
<td align="center">📚<br/><b>Students</b><br/>Learn from cutting-edge research</td>
</tr>
<tr>
<td align="center">🏢<br/><b>Companies</b><br/>Track AI trends & competition</td>
<td align="center">📰<br/><b>Content Creators</b><br/>Find topics for blogs & videos</td>
<td align="center">🤖<br/><b>AI Enthusiasts</b><br/>Explore the latest in AI</td>
</tr>
</table>

---

## ⚡ Quick Start

### 1️⃣ Get Today's Papers (cURL)

```bash
curl https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json
```

### 2️⃣ Python Integration

```python
import requests
import pandas as pd

# Load latest papers
url = "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json"
papers = requests.get(url).json()

# Convert to DataFrame for analysis
df = pd.DataFrame(papers)
print(f"📚 Today's papers: {{len(df)}}")

# Filter by stars
trending = df[df['stars'].astype(int) > 10]
print(f"🔥 Trending papers: {{len(trending)}}")
```

### 3️⃣ JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

async function getTodaysPapers() {{
  const response = await fetch(
    'https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json'
  );
  const papers = await response.json();
  
  console.log(`📚 Found ${{papers.length}} papers today!`);
  papers.forEach(paper => {{
    console.log(`\\n📄 ${{paper.title}}`);
    console.log(`⭐ ${{paper.stars}} stars`);
    console.log(`🔗 ${{paper.details.arxiv_page_url}}`);
  }});
}}

getTodaysPapers();
```

### 4️⃣ Build Your Own Newsletter

```python
import requests
from datetime import datetime

def generate_weekly_digest():
    # Load this week's papers
    papers = requests.get(
        "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/weekly/{week_str}.json"
    ).json()
    
    # Sort by stars
    papers.sort(key=lambda x: int(x['stars']), reverse=True)
    
    # Generate email content
    email = f"# Top AI Papers This Week ({{datetime.now().strftime('%Y-%m-%d')}})\\n\\n"
    
    for i, paper in enumerate(papers[:10], 1):
        email += f"{{i}}. **{{paper['title']}}** ⭐ {{paper['stars']}}\\n"
        email += f"   {{paper['details']['arxiv_page_url']}}\\n\\n"
    
    return email

print(generate_weekly_digest())
```

---

## 📈 Statistics

<table>
<tr>
<td align="center"><b>📄 Today</b><br/><font size="5">{len(papers)}</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">{weekly_count}</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">{monthly_count}</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">{total_papers}+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** {today_display}

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

"""
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'N/A')
        paper_url = paper.get('paper_url', '#')
        authors = paper.get('authors', [])
        stars = paper.get('stars', '0')
        details = paper.get('details', {})
        
        arxiv_url = details.get('arxiv_page_url', '')
        pdf_url = details.get('pdf_url', '')
        github_links = details.get('github_links', [])
        abstract = details.get('abstract', '')
        
        # Handle None or empty abstract
        if not abstract:
            abstract = 'No abstract available.'
        elif len(abstract) > 250:
            abstract = abstract[:247] + "..."
        
        readme += f"""<details>
<summary><b>{i}. {title}</b> ⭐ {stars}</summary>

<br/>

"""
        
        if authors:
            authors_str = ", ".join(authors[:5])
            if len(authors) > 5:
                authors_str += f" _+{len(authors) - 5} more_"
            readme += f"**👥 Authors:** {authors_str}\n\n"
        
        # Links with emojis
        links = []
        if paper_url:
            links.append(f"[🤗 HuggingFace]({paper_url})")
        if arxiv_url:
            links.append(f"[📄 arXiv]({arxiv_url})")
        if pdf_url:
            links.append(f"[📥 PDF]({pdf_url})")
        
        if links:
            readme += f"**🔗 Links:** {' • '.join(links)}\n\n"
        
        # GitHub repos
        if github_links:
            gh_links = " • ".join([f"[⭐ Code]({link})" for link in github_links[:3]])
            readme += f"**💻 Code:** {gh_links}\n\n"
        
        # Abstract in blockquote
        readme += f"> {abstract}\n\n"
        readme += "</details>\n\n"
    
    # Archive section
    readme += f"""---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | {len(papers)} |
| 📅 Today | [`{today_str}.json`](data/daily/{today_str}.json) | {len(papers)} |
| 📆 This Week | [`{week_str}.json`](data/weekly/{week_str}.json) | {weekly_count} |
| 🗓️ This Month | [`{month_str}.json`](data/monthly/{month_str}.json) | {monthly_count} |

### 📜 Recent Days

"""
    
    # List last 7 days in a table
    readme += "| Date | Papers | Link |\n"
    readme += "|------|--------|------|\n"
    
    for i in range(7):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_file = f"data/daily/{date}.json"
        if Path(daily_file).exists():
            with open(daily_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            emoji = "📌" if i == 0 else "📄"
            readme += f"| {emoji} {date} | {count} | [View JSON](data/daily/{date}.json) |\n"
    
    readme += "\n### 📚 Weekly Archives\n\n"
    readme += "| Week | Papers | Link |\n"
    readme += "|------|--------|------|\n"
    
    # List available weeks
    if Path("data/weekly").exists():
        weekly_files = sorted(Path("data/weekly").glob("*.json"), reverse=True)
        for week_file in weekly_files[:4]:  # Last 4 weeks
            week_name = week_file.stem
            with open(week_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            readme += f"| 📅 {week_name} | {count} | [View JSON](data/weekly/{week_name}.json) |\n"
    
    readme += "\n### 🗂️ Monthly Archives\n\n"
    readme += "| Month | Papers | Link |\n"
    readme += "|------|--------|------|\n"
    
    # List available months
    if Path("data/monthly").exists():
        monthly_files = sorted(Path("data/monthly").glob("*.json"), reverse=True)
        for month_file in monthly_files[:6]:  # Last 6 months
            month_name = month_file.stem
            with open(month_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            readme += f"| 🗓️ {month_name} | {count} | [View JSON](data/monthly/{month_name}.json) |\n"
    
    # Features section with enhanced content
    readme += """
---

## ✨ Features

- 🔄 **Automated Daily Updates** - Runs every day at midnight UTC via GitHub Actions
- 📊 **Comprehensive Data** - Full abstracts, author lists, and metadata for every paper
- 🗄️ **Historical Archives** - Daily, weekly, and monthly snapshots for trend analysis
- 🔗 **Direct Links** - Quick access to arXiv, PDFs, GitHub repos, and HuggingFace pages
- 📈 **Trending Papers** - Star counts and popularity metrics to find hot research
- 💾 **JSON Format** - Machine-readable format for easy integration into your projects
- 🎨 **Beautiful README** - Clean, organized presentation updated automatically
- 🔍 **Searchable Archives** - Easy to filter and find papers by date or topic

---

## 🚀 Usage Examples

### View Papers

- **📖 Latest Papers**: Check this README (updated daily at 00:00 UTC)
- **📦 JSON Data**: Download from [`data/latest.json`](data/latest.json)
- **📚 Historical Data**: Browse the [`data/`](data/) directory for archives

### Integrate Into Your Projects

#### Python Example: Daily Digest Script

```python
import requests
from datetime import datetime

# Fetch latest papers
url = "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json"
papers = requests.get(url).json()

# Filter papers with code
papers_with_code = [p for p in papers if p['details'].get('github_links')]

print(f"📊 Papers with code today: {{len(papers_with_code)}}")

for paper in papers_with_code[:5]:
    print(f"\\n📄 {{paper['title']}}")
    print(f"⭐ {{paper['stars']}} stars")
    print(f"🔗 {{paper['details']['arxiv_page_url']}}")
    for repo in paper['details']['github_links'][:1]:
        print(f"💻 {{repo}}")
```

#### Build a Research Tracker

```python
import requests
import json

def track_research_keywords(keywords):
    """Find papers matching your research interests"""
    papers = requests.get(
        "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json"
    ).json()
    
    matches = []
    for paper in papers:
        title = paper.get('title', '').lower()
        abstract = paper.get('details', {}).get('abstract', '').lower()
        
        if any(keyword.lower() in title or keyword.lower() in abstract 
               for keyword in keywords):
            matches.append(paper)
    
    return matches

# Example usage
llm_papers = track_research_keywords(['llm', 'language model', 'gpt', 'transformer'])
print(f"Found {{len(llm_papers)}} papers about LLMs today!")
```

### Use as RSS Alternative

Monitor this repo for daily AI paper updates:
- ⭐ **Star this repository** to show your support
- 👀 **Watch** → Custom → Check "Releases" for notifications
- 🔔 Enable **"All Activity"** to get notified of every daily update
- 📧 Subscribe to GitHub notifications for commits

---

## 📊 Data Structure

```
data/
├── daily/              # Individual day snapshots
│   ├── 2024-12-04.json
│   ├── 2024-12-05.json
│   └── ...
├── weekly/             # Cumulative weekly papers
│   ├── 2024-W48.json
│   └── ...
├── monthly/            # Cumulative monthly papers
│   ├── 2024-12.json
│   └── ...
└── latest.json         # Most recent scrape (always current)
```

### JSON Schema

```json
{{
  "title": "Paper Title",
  "paper_url": "https://huggingface.co/papers/...",
  "authors": ["Author 1", "Author 2"],
  "stars": "42",
  "scraped_date": "2024-12-04",
  "details": {{
    "abstract": "Full paper abstract...",
    "arxiv_page_url": "https://arxiv.org/abs/...",
    "pdf_url": "https://arxiv.org/pdf/...",
    "github_links": ["https://github.com/..."],
    "metadata": {{}},
    "scraped_at": "2024-12-04T00:15:30"
  }}
}}
```

---

## 🛠️ How It Works

This repository uses modern Python tools to provide reliable, automated paper tracking:

**Technology Stack:**
- 🤖 **[Crawl4AI](https://github.com/unclecode/crawl4ai)** - Advanced web scraping framework
- 🍜 **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing and extraction
- ⚙️ **[GitHub Actions](https://github.com/features/actions)** - Automated daily execution
- 🐍 **Python 3.11+** - Data processing and JSON generation

### Daily Workflow

1. 🕐 **Trigger**: GitHub Actions runs at 00:00 UTC daily
2. 🔍 **Scrape**: Fetches HuggingFace Papers trending page
3. 📥 **Extract**: Downloads detailed info for each paper (abstracts, links, metadata)
4. 💾 **Archive**: Saves to daily/weekly/monthly JSON files
5. 📝 **Generate**: Creates this beautiful, updated README
6. ✅ **Commit**: Automatically commits and pushes changes

### Why This Approach?

- ✅ **Reliable**: No manual updates needed, runs automatically
- ✅ **Complete**: Captures full paper details, not just titles
- ✅ **Organized**: Structured archives make trend analysis easy
- ✅ **Accessible**: JSON format works with any programming language
- ✅ **Transparent**: All code is open source and auditable

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs**: [Open an issue](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues) for any problems you find
- 💡 **Feature Requests**: [Share your ideas](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions) for new features
- 🔧 **Code Contributions**: [Submit a PR](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/pulls) with improvements
- 📖 **Documentation**: Help improve README or add examples
- ⭐ **Spread the Word**: Star the repo and share with colleagues!

### Development Setup

```bash
# Clone the repository
git clone https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers.git
cd Daily-HuggingFace-AI-Papers

# Install dependencies
pip install -r requirements.txt

# Run the scraper
python scraper.py
```

### Ideas for Contributions

- Add paper categorization (NLP, CV, RL, etc.)
- Create visualization scripts for trends
- Build a simple search API
- Add RSS feed generation
- Create browser extension
- Add email notification system

---

## 📜 License

MIT License - feel free to use this data for your own projects!

This means you can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ No warranty provided

See [LICENSE](LICENSE) for full details.

---

## 🌟 Star History

If you find this useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=AtharvaDomale/Daily-HuggingFace-AI-Papers&type=Date)](https://star-history.com/#AtharvaDomale/Daily-HuggingFace-AI-Papers&Date)

---

## 💬 Community & Support

### Get Help

- 📖 **Documentation**: Check this README first
- 💬 **Discussions**: [GitHub Discussions](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions) for questions
- 🐛 **Bug Reports**: [Issue Tracker](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues)

### Stay Updated

- ⭐ Star this repo to stay in the loop
- 👀 Watch for new features and updates
- 🐦 Follow [@AtharvaDomale](https://github.com/AtharvaDomale) for project updates

---

## ❓ FAQ

<details>
<summary><b>How often is this updated?</b></summary>
<br/>
Every day at 00:00 UTC via GitHub Actions. You'll always have the latest papers from HuggingFace!
</details>

<details>
<summary><b>Can I use this data in my project?</b></summary>
<br/>
Yes! It's MIT licensed. Use it for research, apps, newsletters, or anything else. Just maintain the license notice.
</details>

<details>
<summary><b>How do I get notified of updates?</b></summary>
<br/>
Star & Watch this repo, or use RSS feeds via GitHub's built-in functionality. You can also write a script to check the JSON daily.
</details>

<details>
<summary><b>Why HuggingFace Papers?</b></summary>
<br/>
HuggingFace Papers curates trending AI research with community engagement (stars). It's a great signal for what's hot in AI research.
</details>

<details>
<summary><b>Can I request specific features?</b></summary>
<br/>
Absolutely! Open a discussion or issue with your ideas. We're always looking to improve!
</details>

---

## 🙏 Acknowledgments

- 🤗 **HuggingFace** for providing the excellent Papers platform
- 🌐 **Crawl4AI** for the robust scraping framework
- 👥 **Contributors** who help improve this project
- ⭐ **Everyone** who stars and uses this repository

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/A
