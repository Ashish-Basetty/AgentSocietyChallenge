import json
import pandas as pd
from tqdm import tqdm
import os
import argparse

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

def get_top_cities_by_users_and_reviews(business_df, review_df):
    """Find the top three cities with the most users and reviews."""
    tqdm.pandas(desc="Calculating Top Cities")
    city_review_counts = business_df.groupby('city').size().sort_values(ascending=False)
    top_cities = city_review_counts.head(3).index.tolist()
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]
    return top_cities, filtered_businesses, filtered_reviews

def save_json(dataframe, output_file):
    """Save a Pandas DataFrame to a JSON file."""
    tqdm.write(f"Saving {output_file}...")
    dataframe.to_json(output_file, orient='records', lines=True)
    tqdm.write(f"{output_file} saved.")

def filter_and_save_data(top_cities, business_df, checkin_df, user_df, tip_df, review_df, output_dir):
    """Filter and save data within the top three cities with progress bars."""
    os.makedirs(output_dir, exist_ok=True)
    tqdm.pandas(desc="Filtering Data for Top Cities")
    
    filtered_businesses = business_df[business_df['city'].isin(top_cities)]
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]
    filtered_users = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])]
    filtered_tips = tip_df[tip_df['business_id'].isin(filtered_businesses['business_id'])]
    filtered_checkins = checkin_df[checkin_df['business_id'].isin(filtered_businesses['business_id'])]

    # Save filtered data in JSON format
    save_json(filtered_businesses, os.path.join(output_dir, 'business.json'))
    save_json(filtered_reviews, os.path.join(output_dir, 'review.json'))
    save_json(filtered_users, os.path.join(output_dir, 'user.json'))
    save_json(filtered_tips, os.path.join(output_dir, 'tip.json'))
    save_json(filtered_checkins, os.path.join(output_dir, 'checkin.json'))

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process the Yelp dataset for analysis.")
    parser.add_argument('--input_dir', required=True, help="Path to the input directory containing Yelp JSON files.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory for saving processed data.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # File paths
    business_file = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    checkin_file = os.path.join(input_dir, 'yelp_academic_dataset_checkin.json')
    user_file = os.path.join(input_dir, 'yelp_academic_dataset_user.json')
    tip_file = os.path.join(input_dir, 'yelp_academic_dataset_tip.json')
    review_file = os.path.join(input_dir, 'yelp_academic_dataset_review.json')

    # Load the datasets with progress bars
    business_df = load_data(business_file)
    checkin_df = load_data(checkin_file)
    user_df = load_data(user_file)
    tip_df = load_data(tip_file)
    review_df = load_data(review_file)

    # Clean the datasets
    business_df = clean_business_data(business_df)
    checkin_df = clean_checkin_data(checkin_df)
    user_df = clean_user_data(user_df)
    tip_df = clean_tip_data(tip_df)
    review_df = clean_review_data(review_df)

    # Find top cities and related data
    top_cities, filtered_businesses, filtered_reviews = get_top_cities_by_users_and_reviews(business_df, review_df)

    print(f"Top 3 cities with the most users and reviews: {top_cities}")

    # Filter and save data for top three cities
    filter_and_save_data(top_cities, business_df, checkin_df, user_df, tip_df, review_df, output_dir)

if __name__ == '__main__':
    main()