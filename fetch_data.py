import json
from pubmed import PubMedRetriever

# --- CONFIGURATION ---
JSON_OUT_PATH = "pubmed_articles_backup.json"
LIMIT_PER_CATEGORY = 60 

CATEGORIES = {
    "Diabetes": "(intermittent fasting OR time-restricted feeding) AND 'Type 2 Diabetes'",
    "Obesity": "(intermittent fasting OR time-restricted eating) AND (obesity OR overweight) AND 'weight loss'",
    "Metabolic": "'alternate day fasting' AND 'metabolic syndrome'",
    "Safety": "(intermittent fasting OR '5:2 diet') AND (safety OR 'adverse effects')",
    "Comparison": "(intermittent fasting) AND ('continuous energy restriction' OR 'standard diet') AND comparison"
}

def format_abstract_text(abstract_dict):
    """Joins abstract parts into a clean narrative."""
    if isinstance(abstract_dict, dict):
        return "\n".join([f"{k}: {v}" for k, v in abstract_dict.items()])
    return str(abstract_dict)

def run_fetch():
    retriever = PubMedRetriever()
    full_articles = []   
    seen_pmids = set()
    stats = {}

    print("="*60)
    print("🚀 MEDI-ASSIST AI: DATA FETCHING PHASE")
    print("="*60)

    for category, query in CATEGORIES.items():
        print(f"\n📂 CATEGORY: {category}")
        pmid_list = retriever.search_pubmed_articles(query, max_results=LIMIT_PER_CATEGORY)
        found_count = len(pmid_list)
        print(f"📑 Found {found_count} matching PMIDs.")
        
        if found_count == 0:
            stats[category] = 0
            continue

        articles = retriever.fetch_pubmed_abstracts(pmid_list)
        new_for_this_category = 0
        
        for art in articles:
            if art['pmid'] in seen_pmids:
                continue
            
            full_abstract = format_abstract_text(art['abstract'])
            
            # Citation logic preserved exactly
            author_names = art['authors'].split(',')
            last_name = author_names[0].split(' ')[-1] if art['authors'] != "No Authors" else "Unknown"
            citation = f"{last_name} et al., {art['publication_date']} (PMID: {art['pmid']})"

            full_articles.append({
                "pmid": art['pmid'],
                "title": art['title'],
                "journal": art['journal'],
                "authors": art['authors'],
                "publication_date": art['publication_date'],
                "category": category,
                "citation": citation,
                "full_abstract": full_abstract
            })
            
            seen_pmids.add(art['pmid'])
            new_for_this_category += 1
        
        stats[category] = new_for_this_category
        print(f"✅ Added {new_for_this_category} unique articles.")

    print("\n💾 Writing master data to:", JSON_OUT_PATH)
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_articles, f, indent=4, ensure_ascii=False)
    
    print(f"✨ Fetching complete. Total unique articles: {len(full_articles)}")

if __name__ == "__main__":
    run_fetch()