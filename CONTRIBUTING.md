# Contributing Guidelines  

## Coding Style 
---  
I want this code to be readable, easy to maintain, and follow best practices widely established in the Python community where possible. Have I done this perfectly? No. But I try. Please try.   
* **Naming of functions**, classes, and variables: help your reader by using descriptive names that can help you to remove redundant comments.  
    * No single letter variables.  
    * Expand acronyms into their full form.  
    * vars and funcs in lower_snake_case, CONSTANTS in UPPER, ClassNames in UpperCamelCase. 
* Don't over comment and create comment drift. Remember, you have 2 things to maintain, the code and the comment. If the code changes, update the comment. 
* Use Python **f-strings** to make code easier to read.  
* Write **doc strings that are descriptive** (I know it takes time, you know what takes more time? Trying to read undocumented code).  
* Put **URLS inside docstrings** to point people to a good article when helpful.  
* **Write tests** to verify work (consider writing doctests even).  
* Never use input() for **anything ever**. If you have to, follow it with .strip() (starting_value = int(input("Please enter a starting value: ").strip()).  
* Absultely use Python **type hints** for function parameters and return values.  
* **Use comprehensions** instead of map(), filter(), reduce() with lambdas.  
* If you need a third party library not in the requirements.txt file, please add it to the requirements.txt file.  
* Strictly use snake_case in file directory for parsing reasons.   
* Make sure your code works before committing it.