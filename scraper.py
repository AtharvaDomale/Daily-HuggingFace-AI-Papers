from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
import json
from datetime import datetime
from pathlib import Path

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


def save_to_json(data, filename="data/papers.json"):
    """Save scraped data to JSON file"""
    Path("data").mkdir(exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(data)} papers to {filename}")


def generate_readme(papers):
    """Generate README.md from scraped papers"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    readme = f"""# 🤖 Daily AI Papers

Automatically updated list of trending AI research papers from HuggingFace.

**Last Updated:** {today}

**Total Papers:** {len(papers)}

---

## 📚 Today's Papers

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
    
    # Footer
    readme += """
## 🔄 Update Schedule

This repository automatically updates daily at 00:00 UTC with the latest AI papers from HuggingFace.

## 📊 Data

All paper data is stored in JSON format in the [`data/papers.json`](data/papers.json) file.

## 🛠️ How It Works

This repository uses:
- **Crawl4AI** for web scraping
- **GitHub Actions** for daily automation
- **Python** for data processing

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
