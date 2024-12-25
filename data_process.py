import json
import pandas as pd
from tqdm import tqdm
import os
import argparse

REQUIRED_FILES_YELP = [
    'yelp_academic_dataset_business.json', 
    'yelp_academic_dataset_user.json', 
    'yelp_academic_dataset_review.json'
]

REQUIRED_FILES_AMAZON = [
    'All_Beauty.jsonl', 
    'meta_ALL_Beauty.jsonl', 
    'Handmade_Products.jsonl', 
    'meta_Handmade_Products.jsonl', 
    'Health_and_Personal_Care.jsonl', 
    'meta_Health_and_Personal_Care.jsonl'
]

REQUIRED_FILES_GOODREADS = [
    'goodreads_books_children.json', 
    'goodreads_reviews_children.json', 
    'goodreads_books_comics_graphic.json', 
    'goodreads_reviews_comics_graphic.json', 
    'goodreads_books_poetry.json', 
    'goodreads_reviews_poetry.json'
]

def load_data(file_path):
    """Load JSON data into a Pandas DataFrame with progress bar."""
    data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=f"Loading {file_path}", unit=" lines"):
            data.append(json.loads(line))
    return pd.DataFrame(data)

def clean_business_data(df):
    """Clean business data by removing invalid entries and splitting categories."""
    tqdm.pandas(desc="Cleaning Business Data")
    df = df.dropna(subset=['business_id', 'categories']).copy()
    df.loc[:, 'categories'] = df['categories'].progress_apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    return df

def clean_checkin_data(df):
    """Clean check-in data by parsing dates."""
    tqdm.pandas(desc="Cleaning Check-in Data")
    df = df.dropna(subset=['business_id', 'date']).copy()
    return df

def clean_user_data(df):
    """Clean user data by removing invalid entries."""
    tqdm.pandas(desc="Cleaning User Data")
    df = df.dropna(subset=['user_id', 'review_count', 'yelping_since']).copy()
    return df

def clean_tip_data(df):
    """Clean tip data by removing invalid entries."""
    tqdm.pandas(desc="Cleaning Tip Data")
    df = df.dropna(subset=['user_id', 'business_id', 'text', 'date']).copy()
    return df

def clean_review_data(df):
    """Clean review data by removing invalid entries."""
    tqdm.pandas(desc="Cleaning Review Data")
    df = df.dropna(subset=['review_id', 'user_id', 'business_id', 'stars', 'text', 'date']).copy()
    return df

def get_top_cities_by_reviews(business_df):
    """Find the top three cities with the most total review counts."""
    tqdm.pandas(desc="Calculating Top Cities")
    city_review_counts = business_df.groupby('city')['review_count'].sum().sort_values(ascending=False)
    top_cities = city_review_counts.head(3).index.tolist()
    return top_cities

def save_json(dataframe, output_file):
    """Save a Pandas DataFrame to a JSON file."""
    tqdm.write(f"Saving {output_file}...")
    dataframe.to_json(output_file, orient='records', lines=True)
    tqdm.write(f"{output_file} saved.")

def filter_data(top_cities, business_df, user_df, review_df):
    """Filter and save data within the top three cities with progress bars."""
    tqdm.pandas(desc="Filtering Data for Top Cities")
    
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]
    filtered_users = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])]

    # Save filtered data in JSON format
    return filtered_businesses, filtered_reviews, filtered_users

def check_required_files(input_dir):
    """Check if all required files exist in the input directory."""
    all_required_files = REQUIRED_FILES_YELP + REQUIRED_FILES_AMAZON + REQUIRED_FILES_GOODREADS
    missing_files = []
    
    for file in all_required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def load_and_process_yelp_data(input_dir):
    # File paths
    business_file = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    user_file = os.path.join(input_dir, 'yelp_academic_dataset_user.json')
    review_file = os.path.join(input_dir, 'yelp_academic_dataset_review.json')

    # Load the datasets with progress bars
    business_df = load_data(business_file)
    user_df = load_data(user_file)
    review_df = load_data(review_file)

    # Clean the datasets
    business_df = clean_business_data(business_df)
    user_df = clean_user_data(user_df)
    review_df = clean_review_data(review_df)

    # Find top cities and related data
    top_cities = get_top_cities_by_reviews(business_df)

    filtered_businesses, filtered_reviews, filtered_users = filter_data(top_cities, business_df, user_df, review_df)
    return filtered_businesses, filtered_reviews, filtered_users

def load_and_process_amazon_data(input_dir):
    """Load and process Amazon dataset."""
    product_files = ['All_Beauty.jsonl', 'Handmade_Products.jsonl', 'Health_and_Personal_Care.jsonl']
    meta_files = ['meta_ALL_Beauty.jsonl', 'meta_Handmade_Products.jsonl', 'meta_Health_and_Personal_Care.jsonl']
    
    all_reviews = pd.concat([load_data(os.path.join(input_dir, f)) for f in product_files])
    all_meta = pd.concat([load_data(os.path.join(input_dir, f)) for f in meta_files])
    
    return all_reviews, all_meta

def load_and_process_goodreads_data(input_dir):
    """Load and process Goodreads dataset."""
    book_files = ['goodreads_books_children.json', 'goodreads_books_comics_graphic.json', 'goodreads_books_poetry.json']
    review_files = ['goodreads_reviews_children.json', 'goodreads_reviews_comics_graphic.json', 'goodreads_reviews_poetry.json']
    
    all_books = pd.concat([load_data(os.path.join(input_dir, f)) for f in book_files])
    all_reviews = pd.concat([load_data(os.path.join(input_dir, f)) for f in review_files])
    
    return all_books, all_reviews

def merge_business_data(yelp_business, amazon_meta, goodreads_books):
    """Merge business data from all sources."""
    # Transform Amazon meta data to business format
    amazon_business = amazon_meta.rename(columns={
        'asin': 'business_id',
        'title': 'name',
        'description': 'attributes',
        'category': 'categories'
    })
    
    # Transform Goodreads books to business format
    goodreads_business = goodreads_books.rename(columns={
        'book_id': 'business_id',
        'title': 'name',
        'description': 'attributes',
        'genres': 'categories'
    })
    
    # Add type column
    yelp_business['type'] = 'location'
    amazon_business['type'] = 'product'
    goodreads_business['type'] = 'book'
    
    return pd.concat([yelp_business, amazon_business, goodreads_business])

def merge_review_data(yelp_reviews, amazon_reviews, goodreads_reviews):
    """Merge review data from all sources."""
    # Standardize column names
    amazon_reviews = amazon_reviews.rename(columns={
        'asin': 'business_id',
        'overall': 'stars',
        'reviewText': 'text',
        'reviewTime': 'date'
    })
    
    goodreads_reviews = goodreads_reviews.rename(columns={
        'book_id': 'business_id',
        'rating': 'stars',
        'review_text': 'text'
    })
    
    # Add source column
    yelp_reviews['source'] = 'yelp'
    amazon_reviews['source'] = 'amazon'
    goodreads_reviews['source'] = 'goodreads'
    
    return pd.concat([yelp_reviews, amazon_reviews, goodreads_reviews])

def create_unified_users(yelp_users, amazon_reviews, goodreads_reviews):
    """Create unified user data including empty users for Amazon and Goodreads."""
    # Create empty users for Amazon
    amazon_users = pd.DataFrame({
        'user_id': amazon_reviews['reviewerID'].unique(),
        'source': 'amazon'
    })
    
    # Create empty users for Goodreads
    goodreads_users = pd.DataFrame({
        'user_id': goodreads_reviews['user_id'].unique(),
        'source': 'goodreads'
    })
    
    # Add source to Yelp users
    yelp_users['source'] = 'yelp'
    
    return pd.concat([yelp_users, amazon_users, goodreads_users])

def main():
    """Main function with updated processing logic."""
    parser = argparse.ArgumentParser(description="Process multiple datasets for analysis.")
    parser.add_argument('--input_dir', required=True, help="Path to the input directory containing all dataset files.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory for saving processed data.")
    args = parser.parse_args()

    # Check required files
    if not check_required_files(args.input_dir):
        return

    # Process Yelp data
    filtered_businesses, filtered_reviews, filtered_users = load_and_process_yelp_data(args.input_dir)
    
    # Process Amazon data
    amazon_reviews, amazon_meta = load_and_process_amazon_data(args.input_dir)
    
    # Process Goodreads data
    goodreads_books, goodreads_reviews = load_and_process_goodreads_data(args.input_dir)
    
    # Merge all data
    merged_business = merge_business_data(filtered_businesses, amazon_meta, goodreads_books)
    merged_reviews = merge_review_data(filtered_reviews, amazon_reviews, goodreads_reviews)
    merged_users = create_unified_users(filtered_users, amazon_reviews, goodreads_reviews)
    
    # Save merged data
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(merged_business, os.path.join(args.output_dir, 'business.json'))
    save_json(merged_reviews, os.path.join(args.output_dir, 'review.json'))
    save_json(merged_users, os.path.join(args.output_dir, 'user.json'))

if __name__ == '__main__':
    main()