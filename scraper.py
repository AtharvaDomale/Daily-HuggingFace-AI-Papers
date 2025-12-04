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
    """Generate README.md with today's papers and archive info"""
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
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
    
    readme = f"""# 🤖 Daily AI Papers

Automatically updated list of trending AI research papers from HuggingFace.

**Last Updated:** {today_str}

## 📊 Statistics

- **Today's Papers:** {len(papers)}
- **This Week:** {weekly_count} papers
- **This Month:** {monthly_count} papers

## 📁 Archives

- **Daily:** [`data/daily/{today_str}.json`](data/daily/{today_str}.json)
- **Weekly:** [`data/weekly/{week_str}.json`](data/weekly/{week_str}.json)
- **Monthly:** [`data/monthly/{month_str}.json`](data/monthly/{month_str}.json)
- **Latest:** [`data/latest.json`](data/latest.json)

---

## 📚 Today's Papers ({today_str})

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
        elif len(abstract) > 300:
            abstract = abstract[:297] + "..."
        
        readme += f"### {i}. {title}\n\n"
        
        if authors:
            authors_str = ", ".join(authors[:5])
            if len(authors) > 5:
                authors_str += f" (+{len(authors) - 5} more)"
            readme += f"**Authors:** {authors_str}\n\n"
        
        readme += f"**⭐ Stars:** {stars}\n\n"
        
        # Links
        links = []
        if paper_url:
            links.append(f"[HuggingFace]({paper_url})")
        if arxiv_url:
            links.append(f"[arXiv]({arxiv_url})")
        if pdf_url:
            links.append(f"[PDF]({pdf_url})")
        
        if links:
            readme += f"**Links:** {' | '.join(links)}\n\n"
        
        # GitHub repos
        if github_links:
            readme += "**GitHub:** "
            readme += " | ".join([f"[Repo]({link})" for link in github_links[:3]])
            readme += "\n\n"
        
        # Abstract
        readme += f"**Abstract:** {abstract}\n\n"
        readme += "---\n\n"
    
    # Archive section
    readme += """
## 📅 Historical Data

### Recent Days
"""
    
    # List last 7 days
    for i in range(7):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_file = f"data/daily/{date}.json"
        if Path(daily_file).exists():
            with open(daily_file, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            readme += f"- **{date}**: [{count} papers](data/daily/{date}.json)\n"
    
    readme += "\n### Weekly Archives\n"
    
    # List available weeks
    weekly_files = sorted(Path("data/weekly").glob("*.json"), reverse=True)
    for week_file in weekly_files[:4]:  # Last 4 weeks
        week_name = week_file.stem
        with open(week_file, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        readme += f"- **{week_name}**: [{count} papers](data/weekly/{week_name}.json)\n"
    
    readme += "\n### Monthly Archives\n"
    
    # List available months
    monthly_files = sorted(Path("data/monthly").glob("*.json"), reverse=True)
    for month_file in monthly_files[:6]:  # Last 6 months
        month_name = month_file.stem
        with open(month_file, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        readme += f"- **{month_name}**: [{count} papers](data/monthly/{month_name}.json)\n"
    
    # Footer
    readme += """

---

## 🔄 Update Schedule

This repository automatically updates daily at 00:00 UTC with the latest AI papers from HuggingFace.

## 🛠️ How It Works

This repository uses:
- **Crawl4AI** for web scraping
- **GitHub Actions** for daily automation
- **Python** for data processing
- **Organized archives** by day, week, and month

## 📊 Data Structure
```
data/
├── daily/          # Individual day snapshots
│   ├── 2024-12-04.json
│   ├── 2024-12-05.json
│   └── ...
├── weekly/         # Cumulative weekly papers
│   ├── 2024-W48.json
│   ├── 2024-W49.json
│   └── ...
├── monthly/        # Cumulative monthly papers
│   ├── 2024-12.json
│   └── ...
└── latest.json     # Always the most recent scrape
```

## 📝 License

MIT License - feel free to use this data for your own projects!

---

*Generated by [Daily AI Papers Bot](https://github.com/yourusername/daily-ai-papers)*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
    print("✅ Generated README.md")


if __name__ == "__main__":
    print("Starting daily AI papers scraper...")
    papers = asyncio.run(scrape_hf_papers())
    save_to_json(papers)
    generate_readme(papers)
    print("\n🎉 All done!")
