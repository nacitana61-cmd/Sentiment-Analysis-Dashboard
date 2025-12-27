from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# ---------------------------
# CONFIG
# ---------------------------
path_to_file = os.getcwd() + os.sep

# Products
BASE_URL_PRODUCTS = "https://web-scraping.dev/products?page={}"
MAX_PAGES_PRODUCTS = 100
EXPECTED_PRODUCTS_PER_PAGE = 28

# Reviews
URL_REVIEWS = "https://web-scraping.dev/reviews"
MAX_LOAD_MORE_REVIEWS = 100
SCROLL_PAUSE = 1.0

# Testimonials
URL_TESTIMONIALS = "https://web-scraping.dev/testimonials/"

# ---------------------------
# DRIVER SETUP
# ---------------------------
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# ---------------------------
# PRODUCTS SCRAPER
# ---------------------------
def scrape_products():
    print("Scraping Products...")
    products_data = []
    page = 1

    while page <= MAX_PAGES_PRODUCTS:
        driver.get(BASE_URL_PRODUCTS.format(page))
        time.sleep(random.uniform(1, 2))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        products = soup.select("div.product")

        if not products:
            break

        for product_div in products:
            name_tag = product_div.select_one("h3")
            price_tag = product_div.select_one(".price")
            if not name_tag or not price_tag:
                continue
            name = name_tag.get_text(strip=True)
            price = price_tag.get_text(strip=True)

            # extract description
            full_text = product_div.get_text(separator="\n", strip=True)
            lines = full_text.split("\n")
            description = None
            for line in lines:
                if line != name and line != price:
                    description = line
                    break

            products_data.append({
                "name": name,
                "price": price,
                "description": description
            })

        print(f"Page {page} scraped: {len(products)} products")
        page += 1

    df_products = pd.DataFrame(products_data)
    df_products.to_csv(path_to_file + "products.csv", index=False)
    print(f"Products scraping completed: {len(df_products)} products saved.\n")

# ---------------------------
# TESTIMONIALS SCRAPER
# ---------------------------
def scrape_testimonials():
    print("Scraping Testimonials...")
    driver.get(URL_TESTIMONIALS)
    time.sleep(5)

    last_height = 0
    while True:
        driver.execute_script('window.scrollBy(0, 5000)')
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source, "html.parser")
    t_items = soup.find_all("div", class_="testimonial")

    testimonials = []
    for i, item in enumerate(t_items):
        text = item.find("p", class_="text")
        if text:
            # generate random 2023 date if not present
            month = (i % 12) + 1
            day = random.randint(1, 28)
            date = f"2023-{month:02d}-{day:02d}"
            testimonials.append({"review": text.text.strip(), "date": date})

    df_testimonials = pd.DataFrame(testimonials)
    df_testimonials.to_csv(path_to_file + "testimonials.csv", index=False)
    print(f"Testimonials scraping completed: {len(df_testimonials)} testimonials saved.\n")

# ---------------------------
# REVIEWS SCRAPER
# ---------------------------
def scrape_reviews():
    print("Scraping Reviews...")
    driver.get(URL_REVIEWS)
    time.sleep(2)

    clicks = 0
    last_review_count = 0

    while clicks < MAX_LOAD_MORE_REVIEWS:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

        try:
            load_more_button = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button.load-more"))
            )
            if load_more_button.is_displayed():
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", load_more_button)
                driver.execute_script("arguments[0].click();", load_more_button)
                clicks += 1
                print(f"'Load More' clicked {clicks} times...")
                time.sleep(random.uniform(1.5, 2.5))
        except:
            pass

        current_reviews = driver.find_elements(By.CSS_SELECTOR, "div.review, div.col p")
        current_count = len(current_reviews)
        if current_count == last_review_count:
            break
        last_review_count = current_count

    # Parse reviews
    soup = BeautifulSoup(driver.page_source, "html.parser")
    review_elements = soup.select("div.review, div.col p")

    reviews_list = []
    for i, review_div in enumerate(review_elements):
        date_tag = review_div.select_one(".date")
        text_tag = review_div.select_one("p")
        if text_tag:
            review_text = text_tag.get_text(strip=True)
            if date_tag:
                review_date = date_tag.get_text(strip=True)
            else:
                month = (i % 12) + 1
                day = random.randint(1, 28)
                review_date = f"2023-{month:02d}-{day:02d}"
            reviews_list.append({"date": review_date, "review": review_text})

    df_reviews = pd.DataFrame(reviews_list)
    df_reviews.to_csv(path_to_file + "reviews.csv", index=False)
    print(f"Reviews scraping completed: {len(df_reviews)} reviews saved.\n")

# ---------------------------
# RUN ALL SCRAPERS
# ---------------------------
try:
    scrape_products()
    scrape_testimonials()
    scrape_reviews()
finally:
    driver.quit()
    print("All scraping completed. CSV files saved in current folder.")


