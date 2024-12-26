# Data Preparation Guide

## Main Workflow

1. Download the dataset from the link provided in the table below.
2. Unzip the dataset and put it in the **Any** directory.
    - Make sure all data files are in the same directory.
    - Necessary files for Yelp:
        - `yelp_academic_dataset_review.json`
        - `yelp_academic_dataset_business.json`
        - `yelp_academic_dataset_user.json`
    - Necessary files for Amazon:
        - `All_Beauty.jsonl`
        - `meta_All_Beauty.jsonl`
        - `Handmade_Products.jsonl`
        - `meta_Handmade_Products.jsonl`
        - `Health_and_Personal_Care.jsonl`
        - `meta_Health_and_Personal_Care.jsonl`
    - Necessary files for Goodreads:
        - `goodreads_reviews_comics_graphic.json`
        - `goodreads_books_comics_graphic.json`
        - `goodreads_reviews_poetry.json`
        - `goodreads_books_poetry.json`
3. Run the `data_preparation.py` script to prepare the data for the simulation and recommendation tasks.
```bash
python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
```

## Dataset Overview and Download Links

|                                       | len(review)   | len(business) | len(user)   | link                                                         |
| ------------------------------------- | ------------- | ------------- | ----------- | ------------------------------------------------------------ |
| **Yelp**                              | **-** | **-**    | **-** | [download](https://www.yelp.com/dataset)                                                             |
| -                                     | -             | -             | -           |                                                              |
| Amazon<br />_All_Beauty               | 701,500       | 112,600       | 632,000     | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz) <br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz) |
| Amazon<br />_Handmade_Products        | 664,200       | 164,700       | 664,200     | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Handmade_Products.jsonl.gz)<br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Handmade_Products.jsonl.gz) |
| Amazon<br />_Health_and_Personal_Care | 494,100       | 60,300        | 494,100     | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Health_and_Personal_Care.jsonl.gz)<br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Health_and_Personal_Care.jsonl.gz) |
| **Amazon**                            | **1,859,800** | **337,600**   |             |                                                              |
| -                                     | -             | -             | -           |                                                              |
| Goodreads<br />_Children              | 734,640       | 124,082       |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz)<br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_children.json.gz) |
| Goodreads<br />_Comics&Graphic        | 542,338       | 89,411        |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz)<br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz) |
| Goodreads<br />_Poetry                | 154,555       | 36,514        |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz)<br />[meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz) |
| **Goodreads**                         | **1,431,533** | **250,007**   |             |                                                              |