import json
import logging
import uuid
import pandas as pd
from tqdm import tqdm
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FILES_YELP = [
    'yelp_academic_dataset_business.json', 
    'yelp_academic_dataset_user.json', 
    'yelp_academic_dataset_review.json'
]

REQUIRED_FILES_AMAZON = [
    'Industrial_and_Scientific.csv', 
    'Musical_Instruments.csv', 
    'Video_Games.csv',
    'Industrial_and_Scientific.jsonl', 
    'Musical_Instruments.jsonl', 
    'Video_Games.jsonl',
    'meta_Industrial_and_Scientific.jsonl', 
    'meta_Musical_Instruments.jsonl', 
    'meta_Video_Games.jsonl'
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

def save_json(dataframe, output_file):
    """Save a Pandas DataFrame to a JSON file."""
    logging.info(f"Saving {output_file}...")
    dataframe.to_json(output_file, orient='records', lines=True)
    logging.info(f"{output_file} saved.")

def filter_data(top_cities, business_df, user_df, review_df):
    """Filter and save data within the top three cities with progress bars."""
    logging.info("Filtering data for top cities...")
    
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]
    filtered_users = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])]

    # Save filtered data in JSON format
    return filtered_businesses, filtered_reviews, filtered_users

def check_required_files(input_dir):
    """Check if all required files exist in the input directory."""
    all_required_files = REQUIRED_FILES_YELP
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
    logging.info("Loading and processing Yelp data...")
    # File paths
    business_file = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    user_file = os.path.join(input_dir, 'yelp_academic_dataset_user.json')
    review_file = os.path.join(input_dir, 'yelp_academic_dataset_review.json')

    # Find top cities and related data
    logging.info("Finding top cities by reviews for Yelp...")
    top_cities = ['Philadelphia']

    # Load businesses first and filter
    logging.info("Loading businesses...")
    business_df = load_data(business_file)
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    philly_business_ids = set(filtered_businesses['business_id'].unique())
    logging.info(f"Found {len(philly_business_ids)} businesses in {top_cities}")
    
    # Stream reviews and filter in memory-efficient way
    logging.info("Streaming and filtering reviews...")
    filtered_reviews_data = []
    with open(review_file, 'r') as file:
        for line in tqdm(file, desc=f"Filtering reviews", unit=" lines"):
            review = json.loads(line)
            if review.get('business_id') in philly_business_ids:
                filtered_reviews_data.append(review)
    filtered_reviews = pd.DataFrame(filtered_reviews_data)
    logging.info(f"Found {len(filtered_reviews)} reviews for selected cities businesses")
    
    # Get unique user IDs from filtered reviews
    philly_user_ids = set(filtered_reviews['user_id'].unique())
    
    # Stream users and filter
    logging.info("Streaming and filtering users...")
    filtered_users_data = []
    with open(user_file, 'r') as file:
        for line in tqdm(file, desc=f"Filtering users", unit=" lines"):
            user = json.loads(line)
            if user.get('user_id') in philly_user_ids:
                filtered_users_data.append(user)
    filtered_users = pd.DataFrame(filtered_users_data)
    logging.info(f"Found {len(filtered_users)} users with reviews in selected cities")
    
    return filtered_businesses, filtered_reviews, filtered_users

def merge_business_data(yelp_business, output_file=None):
    """Merge business data from all sources while preserving source-specific columns."""
    logging.info("Merging business data for business...")
    
    yelp_business = yelp_business.rename(columns={
        'business_id': 'item_id',
    })
    yelp_business['source'] = 'yelp'
    yelp_business['type'] = 'business'
    yelp_json = json.loads(yelp_business.to_json(orient='records'))
    
    merged_json = yelp_json
    
    if output_file:
        logging.info(f"Saving merged business data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

def merge_review_data(yelp_reviews, output_file=None):
    """Merge review data from all sources while preserving source-specific columns."""
    logging.info("Merging review data for reviews...")
    
    # 将Yelp评论转换为json格式
    yelp_reviews = yelp_reviews.rename(columns={
        'business_id': 'item_id',
    })
    yelp_reviews['source'] = 'yelp'
    yelp_reviews['type'] = 'business'
    yelp_json = json.loads(yelp_reviews.to_json(orient='records'))
    
    merged_json = yelp_json
    
    if output_file:
        logging.info(f"Saving merged review data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

def create_unified_users(yelp_users, output_file=None):
    """Create unified user data while preserving Yelp-specific user information."""
    logging.info("Merging users...")
    
    # 将Yelp用户数据转换为json格式
    yelp_users['source'] = 'yelp'
    yelp_json = json.loads(yelp_users.to_json(orient='records'))
    
    # 合并所有json数据
    merged_json = yelp_json
    
    # 如果指定了输出文件，则保存
    if output_file:
        logging.info(f"Saving merged user data to {output_file}...")
        with open(output_file, 'w') as f:
            for item in merged_json:
                f.write(json.dumps(item) + '\n')

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
    # amazon_reviews, amazon_meta = load_and_process_amazon_data(args.input_dir)
    
    # Process Goodreads data
    # goodreads_books, goodreads_reviews = load_and_process_goodreads_data(args.input_dir)
    
    # Merge all data
    os.makedirs(args.output_dir, exist_ok=True)
    merge_business_data(filtered_businesses, os.path.join(args.output_dir, 'item.json'))
    merge_review_data(filtered_reviews, os.path.join(args.output_dir, 'review.json'))
    create_unified_users(filtered_users, os.path.join(args.output_dir, 'user.json'))

    logging.info("Data processing completed successfully.")

if __name__ == '__main__':
    main()