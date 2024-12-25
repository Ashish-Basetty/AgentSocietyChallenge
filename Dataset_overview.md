|                                       | len(review)   | len(business) | len(user)   | link                                                         |
| ------------------------------------- | ------------- | ------------- | ----------- | ------------------------------------------------------------ |
| **Yelp**                              | **1,827,165** | **32,846**    | **558,062** |                                                              |
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





```json
download_links = [
     {
         dataset = "Amazon",
         category = "All_Beauty",
         review = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
         meta = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"
     },
    {
        dataset = "Amazon",
        category = "Handmade_Products",
        review = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Handmade_Products.jsonl.gz",
        meta = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Handmade_Products.jsonl.gz"
    },
    {
        dataset = "Amazon",
        category = "Health_and_Personal_Care",
        review = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Health_and_Personal_Care.jsonl.gz",
        meta = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Health_and_Personal_Care.jsonl.gz"
    },
    {
        dataset = "Goodereads",
        category = "Children",
        review = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
        meta = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_children.json.gz"
    },
    {
        dataset = "Goodreads",
        category = "Comics&Graphic",
        review = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
        meta = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz"
    },
    {
        dataset = "Goodreads",
        category = "Poetry",
        review = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
        meta = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz"
    }
]
```





数据说明：

Yelp Dataset: [Yelp Dataset](https://www.yelp.com/dataset)

Track1-simulation：
（1）Amazon product review dataset：https://amazon-reviews-2023.github.io/   包含用户（user_id字段）对商品（asin字段）的评论（text字段）、评分（rating字段）、有帮助程度（helpful_votes，类似于之前的useful）
（2）Goodreads：https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home#h.p_VCP_qovwtnn1（仅用来看数据集信息，里面下载链接失效了，在这下载：https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/），包含用户对各类图书的review和rating，以及其他用户的评论数（n_comments）和支持数（n_votes）

Track2-recommendation：
（1）Amazon product review dataset（跟上面track1的来自同一个数据集，但是专门有一个rating only的用于推荐的benchmark）：https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only，这里面是至少包含五条交互的用户评分记录，可以把评分>3的认为是正样本，未交互过的采样当负样本。
（2）Goodreads（跟track数据集一样）：https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home#h.p_VCP_qovwtnn1，自行提取用户rating数据，按评分处理正负样本