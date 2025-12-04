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
    """Generate an attractive README.md with better design"""
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

# 🤖 Daily AI Papers

### 📊 Trending AI Research Papers from HuggingFace

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/yourusername/daily-ai-papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-{len(papers)}-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-{total_papers}+-orange?style=for-the-badge&logo=academia)](data/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Automatically updated every day at 00:00 UTC** ⏰

[📊 View Data](data/) | [🔍 Latest Papers](data/latest.json) | [📅 Archives](#-historical-archives) | [⭐ Star This Repo](#)

</div>

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
    monthly_files = sorted(Path("data/monthly").glob("*.json"), reverse=True)
    for month_file in monthly_files[:6]:  # Last 6 months
        month_name = month_file.stem
        with open(month_file, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        readme += f"| 🗓️ {month_name} | {count} | [View JSON](data/monthly/{month_name}.json) |\n"
    
    # Features section
    readme += """
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
response = requests.get('https://raw.githubusercontent.com/yourusername/daily-ai-papers/main/data/latest.json')
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

- 🐛 [Report Issues](https://github.com/yourusername/daily-ai-papers/issues)
- 💡 [Submit Ideas](https://github.com/yourusername/daily-ai-papers/discussions)
- 🔧 [Pull Requests Welcome](https://github.com/yourusername/daily-ai-papers/pulls)

---

## 📜 License

MIT License - feel free to use this data for your own projects!

See [LICENSE](LICENSE) for more details.

---

## 🌟 Star History

If you find this useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/daily-ai-papers&type=Date)](https://star-history.com/#yourusername/daily-ai-papers&Date)

---

## 📬 Contact & Support

- 💬 [GitHub Discussions](https://github.com/yourusername/daily-ai-papers/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/daily-ai-papers/issues)
- ⭐ Don't forget to star this repo!

---

<div align="center">

**Made with ❤️ by the AI Community**

[⬆ Back to Top](#-daily-ai-papers)

</div>
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
    print("✅ Generated enhanced README.md")


if __name__ == "__main__":
    print("Starting daily AI papers scraper...")
    papers = asyncio.run(scrape_hf_papers())
    save_to_json(papers)
    generate_readme(papers)
    print("\n🎉 All done!")
