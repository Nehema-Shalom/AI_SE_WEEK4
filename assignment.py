#TASK 1
#MANUAL IMPLEMENTATION
def sort_dicts_manual(data, key):
    """Return a new sorted list by key (stable)."""
    return sorted(data, key=lambda d: d[key])

# Example usage
sample = [{'name':'a','age':30},{'name':'b','age':25},{'name':'c','age':28}]
print('Manual sort ->', sort_dicts_manual(sample,'age'))

# AI-suggested in-place approach (example)
def sort_dicts_ai(data, key):
    # sorts in-place and returns the same list
    data.sort(key=lambda d: d.get(key))
    return data

print('AI sort ->', sort_dicts_ai(sample.copy(),'age'))
"""
Both functions achieve sorting by key, but Copilot’s version modifies the list in-place, saving memory and slightly improving efficiency for large datasets. 
Manual implementation, while explicit, creates a new sorted list, which can double memory usage. 
Copilot’s approach is faster but assumes all keys exist — introducing risk of runtime errors. Overall, Copilot accelerates development by providing working patterns instantly, but developers must review AI-generated code for edge cases.
The best approach combines AI assistance with human validation for clean, maintainable results.
 """


#TASK 2
#automated testing with selenium
# selenium_login_test.py (run locally on your machine)
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def run_login_test():
    driver = webdriver.Chrome()  # ensure chromedriver is installed and on PATH
    try:
        driver.get('https://example.com/login')
        time.sleep(1)
        driver.find_element(By.ID, 'username').send_keys('valid_user')
        driver.find_element(By.ID, 'password').send_keys('correct_password')
        driver.find_element(By.ID, 'submit').click()
        time.sleep(2)
        assert 'Dashboard' in driver.title
        print('Login test passed')
    except Exception as e:
        print('Login test failed', e)
    finally:
        driver.quit()
#TASK 3
#predictive analytics for resource allocation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
y = pd.Series(bc.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))