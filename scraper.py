from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://huggingface.co"


async def scrape_paper_details(crawler, paper_url):
    """Scrape individual paper page for detailed information"""
    try:
        result = await crawler.arun(url=paper_url)
        soup = BeautifulSoup(result.html, "html.parser")

        title = soup.select_one("h1")
        title = title.get_text(strip=True) if title else None

        abstract_block = soup.select_one("div.prose")
        abstract = abstract_block.get_text(" ", strip=True) if abstract_block else None

        arxiv_page_url = None
        pdf_url = None
        
        for a in soup.select("a[href*='arxiv.org']"):
            href = a.get("href", "")
            if "/abs/" in href:
                arxiv_page_url = href
            elif "/pdf/" in href or href.endswith(".pdf"):
                pdf_url = href

        github_links = list(set([
            a.get("href") for a in soup.select("a[href*='github.com']")
            if a.get("href")
        ]))

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
            
            title_tag = card.select_one("h3 a")
            title = title_tag.get_text(strip=True) if title_tag else None
            paper_url = BASE_URL + title_tag.get("href") if title_tag else None
            
            if not paper_url:
                print("  ⚠️  No URL found, skipping")
                continue

            print(f"  Title: {title}")

            authors = [a.get("title") for a in card.select("ul li[title]") if a.get("title")]

            star_tag = card.select_one("a.flex span")
            stars = star_tag.get_text(strip=True) if star_tag else "0"

            print(f"  Fetching details from {paper_url}...")
            details = await scrape_paper_details(crawler, paper_url)

            paper_data = {
                "title": title,
                "paper_url": paper_url,
                "authors": authors,
                "stars": stars,
                "details": details
            }
            
            all_papers.append(paper_data)
            
            if idx < len(cards):
                await asyncio.sleep(1.0)

        return all_papers


def save_to_json(data):
    """Save scraped data with daily, weekly, and monthly archives"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    week_str = today.strftime("%Y-W%W")
    month_str = today.strftime("%Y-%m")
    
    Path("data/daily").mkdir(parents=True, exist_ok=True)
    Path("data/weekly").mkdir(parents=True, exist_ok=True)
    Path("data/monthly").mkdir(parents=True, exist_ok=True)
    
    for paper in data:
        paper["scraped_date"] = today_str
    
    daily_file = f"data/daily/{today_str}.json"
    with open(daily_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved daily snapshot: {daily_file}")
    
    weekly_file = f"data/weekly/{week_str}.json"
    weekly_data = []
    if Path(weekly_file).exists():
        with open(weekly_file, "r", encoding="utf-8") as f:
            weekly_data = json.load(f)
    weekly_data.extend(data)
    with open(weekly_file, "w", encoding="utf-8") as f:
        json.dump(weekly_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated weekly archive: {weekly_file}")
    
    monthly_file = f"data/monthly/{month_str}.json"
    monthly_data = []
    if Path(monthly_file).exists():
        with open(monthly_file, "r", encoding="utf-8") as f:
            monthly_data = json.load(f)
    monthly_data.extend(data)
    with open(monthly_file, "w", encoding="utf-8") as f:
        json.dump(monthly_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated monthly archive: {monthly_file}")
    
    with open("data/latest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved latest papers: data/latest.json")


def generate_readme(papers):
    """Generate enhanced README with all improvements"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    today_display = today.strftime("%B %d, %Y")
    week_str = today.strftime("%Y-W%W")
    month_str = today.strftime("%Y-%m")
    
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
    
    total_papers = 0
    if Path("data/daily").exists():
        for daily_file in Path("data/daily").glob("*.json"):
            with open(daily_file, "r", encoding="utf-8") as f:
                total_papers += len(json.load(f))
    
    # START README GENERATION
    readme_content = generate_readme_content(
        papers, len(papers), weekly_count, monthly_count, 
        total_papers, today_display, today_str, week_str, month_str, today
    )
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Generated enhanced README.md")


def generate_readme_content(papers, papers_count, weekly_count, monthly_count, 
                            total_papers, today_display, today_str, week_str, month_str, today):
    """Generate the complete README content"""
    
    content = f"""<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-{papers_count}-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
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

---

## 📈 Statistics

<table>
<tr>
<td align="center"><b>📄 Today</b><br/><font size="5">{papers_count}</font><br/>papers</td>
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
    
    # Add papers
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
        
        if not abstract:
            abstract = 'No abstract available.'
        elif len(abstract) > 250:
            abstract = abstract[:247] + "..."
        
        content += f"""<details>
<summary><b>{i}. {title}</b> ⭐ {stars}</summary>

<br/>

"""
        
        if authors:
            authors_str = ", ".join(authors[:5])
            if len(authors) > 5:
                authors_str += f" _+{len(authors) - 5} more_"
            content += f"**👥 Authors:** {authors_str}\n\n"
        
        links = []
        if paper_url:
            links.append(f"[🤗 HuggingFace]({paper_url})")
        if arxiv_url:
            links.append(f"[📄 arXiv]({arxiv_url})")
        if pdf_url:
            links.append(f"[📥 PDF]({pdf_url})")
        
        if links:
            content += f"**🔗 Links:** {' • '.join(links)}\n\n"
        
        if github_links:
            gh_links = " • ".join([f"[⭐ Code]({link})" for link in github_links[:3]])
            content += f"**💻 Code:** {gh_links}\n\n"
        
        content += f"> {abstract}\n\n"
        content += "</details>\n\n"
    
    # Add rest of README
    content += generate_archives_section(today_str, week_str, month_str, papers_count, weekly_count, monthly_count, today)
    content += generate_features_section()
    content += generate_footer_section()
    
    return content


def generate_archives_section(today_str, week_str, month_str, papers_count, weekly_count, monthly_count, today):
    """Generate archives section"""
    section = f"""---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | {papers_count} |
| 📅 Today | [`{today_str}.json`](data/daily/{today_str}.json) | {papers_count} |
| 📆 This Week | [`{week_str}.json`](data/weekly/{week_str}.json) | {weekly_count} |
| 🗓️ This Month | [`{month_str}.json`](data/monthly/{month_str}.json) | {monthly_count} |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
"""
    
    for i in range(7):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_file = f"data/daily/{date}.json"
        if Path(daily_file).exists():
            with open(daily_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            emoji = "📌" if i == 0 else "📄"
            section += f"| {emoji} {date} | {count} | [View JSON](data/daily/{date}.json) |\n"
    
    section += "\n### 📚 Weekly Archives\n\n| Week | Papers | Link |\n|------|--------|------|\n"
    
    if Path("data/weekly").exists():
        weekly_files = sorted(Path("data/weekly").glob("*.json"), reverse=True)
        for week_file in weekly_files[:4]:
            week_name = week_file.stem
            with open(week_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            section += f"| 📅 {week_name} | {count} | [View JSON](data/weekly/{week_name}.json) |\n"
    
    section += "\n### 🗂️ Monthly Archives\n\n| Month | Papers | Link |\n|------|--------|------|\n"
    
    if Path("data/monthly").exists():
        monthly_files = sorted(Path("data/monthly").glob("*.json"), reverse=True)
        for month_file in monthly_files[:6]:
            month_name = month_file.stem
            with open(month_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            section += f"| 🗓️ {month_name} | {count} | [View JSON](data/monthly/{month_name}.json) |\n"
    
    return section


def generate_features_section():
    """Generate features and usage sections"""
    return """
---

## ✨ Features

- 🔄 **Automated Daily Updates** - Runs every day at midnight UTC
- 📊 **Comprehensive Data** - Abstracts, authors, links, and metadata
- 🗄️ **Historical Archives** - Daily, weekly, and monthly snapshots
- 🔗 **Direct Links** - arXiv, PDF, GitHub repos, and HuggingFace pages
- 📈 **Trending Papers** - Star counts and popularity metrics
- 💾 **JSON Format** - Easy to parse and integrate into your projects
- 🎨 **Clean Interface** - Beautiful, organized README

---

## 🚀 Usage

### View Papers

- **Latest Papers**: Check this README (updated daily)
- **JSON Data**: Download from [`data/latest.json`](data/latest.json)
- **Historical Data**: Browse the [`data/`](data/) directory

### Integrate Into Your Project

```python
import requests

# Get latest papers
response = requests.get('https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json')
papers = response.json()

for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"arXiv: {paper['details']['arxiv_page_url']}")
    print(f"PDF: {paper['details']['pdf_url']}")
```

### Use as RSS Alternative

Monitor this repo for daily AI paper updates:
- ⭐ Star this repository
- 👀 Watch for notifications
- 🔔 Enable "All Activity" for daily updates

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
└── latest.json         # Most recent scrape
```

### JSON Schema

```json
{
  "title": "Paper Title",
  "paper_url": "https://huggingface.co/papers/...",
  "authors": ["Author 1", "Author 2"],
  "stars": "42",
  "scraped_date": "2024-12-04",
  "details": {
    "abstract": "Paper abstract...",
    "arxiv_page_url": "https://arxiv.org/abs/...",
    "pdf_url": "https://arxiv.org/pdf/...",
    "github_links": ["https://github.com/..."],
    "metadata": {}
  }
}
```

---

## 🛠️ How It Works

This repository uses:

- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** - Modern web scraping framework
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing
- **[GitHub Actions](https://github.com/features/actions)** - Automated daily runs
- **Python 3.11+** - Data processing and generation

### Workflow

1. 🕐 GitHub Actions triggers at 00:00 UTC daily
2. 🔍 Scrapes HuggingFace Papers page
3. 📥 Downloads detailed info for each paper
4. 💾 Saves to daily/weekly/monthly archives
5. 📝 Generates this beautiful README
6. ✅ Commits and pushes updates

---

## 🤝 Contributing

Found a bug or have a feature request? 

- 🐛 [Report Issues](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues)
- 💡 [Submit Ideas](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions)
- 🔧 [Pull Requests Welcome](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/pulls)

"""


def generate_footer_section():
    """Generate footer section"""
    return """---

## 📜 License

MIT License - feel free to use this data for your own projects!

See [LICENSE](LICENSE) for more details.

---

## 🌟 Star History

If you find this useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=AtharvaDomale/Daily-HuggingFace-AI-Papers&type=Date)](https://star-history.com/#AtharvaDomale/Daily-HuggingFace-AI-Papers&Date)

---

## 📬 Contact & Support

- 💬 [GitHub Discussions](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions)
- 🐛 [Issue Tracker](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues)
- ⭐ Don't forget to star this repo!

---

<div align="center">

**Made with ❤️ for the AI Community**

[⬆ Back to Top](#-daily-huggingface-ai-papers)

</div>
"""


if __name__ == "__main__":
    print("Starting daily AI papers scraper...")
    papers = asyncio.run(scrape_hf_papers())
    save_to_json(papers)
    generate_readme(papers)
    print("\n🎉 All done!")
