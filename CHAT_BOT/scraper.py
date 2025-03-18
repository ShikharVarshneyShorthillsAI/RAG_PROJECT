import json
from selenium.common.exceptions import WebDriverException, NoSuchElementException, TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
import string
import time
import os

class SeleniumScraper:
    def __init__(self, url):
        """Initialize the WebDriver and open the given URL."""
        self.scraper = webdriver.Chrome()
        self.url = url
        self.failed_urls = []
        self.scraped_data = {}
        os.makedirs("scraped_data", exist_ok=True)

    def get_disease_data(self, elements):
        data = {}
        key, value = "", ""
        
        for element in elements:
            tag = element.tag_name.lower()
            if tag not in ["h2", "h3", "p", "ul"]:
                continue
            
            if tag in ["h2", "h3"]:
                if value:
                    data[key] = value.strip()
                    value = ""
                key = element.text
            elif tag == "p":
                value += element.text + " "
            elif tag == "ul":
                value += " ".join([j.text for j in element.find_elements(By.TAG_NAME, "li")]) + " "
        
        if value:
            data[key] = value.strip()
        return data

    def scrap_it(self):
        scraper = self.scraper
        try:
            scraper.get(self.url)
        except (WebDriverException, TimeoutException) as e:
            print(f"Failed to load {self.url}: {e}")
            return
        
        a_tags = scraper.find_elements(By.TAG_NAME, "a")
        alphabets = list(string.ascii_uppercase)
        it, alphabet_tags = 0, []
        
        for i in a_tags:
            if alphabets[it] == 'Q':  # Skip 'Q' if not present
                it += 1
            if i.text == alphabets[it]:
                alphabet_tags.append(i)
                it += 1
                if it == 26:
                    break
        
        for i in alphabet_tags:
            print(f"Processing Alphabet: {i.text}")
            try:
                i.click()
                time.sleep(4)
            except WebDriverException as e:
                print(f"Error clicking alphabet {i.text}: {e}")
                continue
            
            a_tags = scraper.find_elements(By.TAG_NAME, "a")
            disease_tags = [i for i in a_tags if i.get_attribute("href") and 
                            i.get_attribute("href").startswith("https://www.mayoclinic.org/diseases-conditions/") and 
                            i.get_attribute("href")[-3].isdigit()]
            
            for disease in disease_tags:
                url = disease.get_attribute("href")
                disease_name = url.split("/")[4].replace("-", " ")
                print(f"Opening disease page: {disease_name}")
                
                try:
                    scraper.execute_script("window.open(arguments[0], '_blank');", url)
                    time.sleep(2)
                    scraper.switch_to.window(scraper.window_handles[-1])
                    
                    # Try to get overview data from multiple XPaths
                    overview_data = {}
                    for xpath in ['//*[@id="main-content"]/div[1]/div[1]/div[2]', '//*[@id="main-content"]/div[1]/div[1]/div[3]']:
                        try:
                            main_content = scraper.find_element(By.XPATH, xpath)
                            elements = main_content.find_elements(By.XPATH, "./*")
                            overview_data = self.get_disease_data(elements)
                            if "Symptoms" in overview_data:
                                break
                        except NoSuchElementException:
                            continue
                    
                    if "Symptoms" not in overview_data:
                        print(f"Skipping {disease_name}: Symptoms data not found")
                        scraper.close()
                        if len(scraper.window_handles) > 0:
                            scraper.switch_to.window(scraper.window_handles[0])
                        continue
                    
                    diagnosis_data = {}
                    diagnosis_tag = next((i for i in scraper.find_elements(By.TAG_NAME, "a") if i.text == "Diagnosis & treatment"), None)
                    
                    if diagnosis_tag:
                        try:
                            scraper.execute_script("window.open(arguments[0], '_blank');", diagnosis_tag.get_attribute("href"))
                            time.sleep(5)
                            scraper.switch_to.window(scraper.window_handles[-1])
                            for xpath in ['//*[@id="main-content"]/div[1]/div[1]/div[2]', '//*[@id="main-content"]/div[1]/div[1]/div[3]']:
                                try:
                                    diagnosis_main_content = scraper.find_element(By.XPATH, xpath)
                                    child_elements = diagnosis_main_content.find_elements(By.XPATH, "./*")
                                    diagnosis_data = self.get_disease_data(child_elements)
                                    break
                                except NoSuchElementException:
                                    continue
                            scraper.close()
                            if len(scraper.window_handles) > 0:
                                scraper.switch_to.window(scraper.window_handles[-1])
                        except (NoSuchElementException, TimeoutException):
                            print(f"Diagnosis and Treatment section not found for {disease_name}")
                    
                    self.scraped_data[url] = {"Overview": overview_data, "Diagnosis and Treatment": diagnosis_data}
                    with open(f"scraped_data/{disease_name}.json", "w") as f:
                        json.dump(self.scraped_data[url], f, indent=4)
                    print(f"Data saved for {disease_name}")
                
                except (WebDriverException, TimeoutException) as e:
                    print(f"Error scraping {url}: {e}")
                    self.failed_urls.append(url)
                
                finally:
                    # Close the disease tab safely
                    if len(scraper.window_handles) > 1:
                        scraper.close()
                        try:
                            scraper.switch_to.window(scraper.window_handles[0])
                        except IndexError:
                            print("Base window is closed unexpectedly.")
                    else:
                        print("No other windows open. Skipping window switch.")
            
            scraper.back()
            break

    def close_chrome(self):
        time.sleep(5)
        self.scraper.quit()
        
        if self.failed_urls:
            with open("failed_urls.txt", "w") as f:
                for url in self.failed_urls:
                    f.write(url + "\n")
        
        with open("full_scraped_data.json", "w") as f:
            json.dump(self.scraped_data, f, indent=4)

if __name__ == "__main__":
    url = "https://www.mayoclinic.org/"
    scraper = SeleniumScraper(url)
    scraper.scrap_it()
    scraper.close_chrome()
