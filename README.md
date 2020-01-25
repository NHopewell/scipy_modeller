The primary contact for this for this project is Nick Hopewell  
<Nicholas.Hopewell@cic.gc.ca>    


# scipy-modeler  
--- 
An overview of some applications and goals of scipy-modeler
---
The scipy-modeler library is a Data Science library which supports data preprocessing and predictive modelling by providing methods for data input, wrangling, and modelling which every data scientist must write at some point no matter what. This library also generates a ML transformers and constructs them into data pipelines behind the scenes. Additionally, scipy-modeller also lets the user persist data pipelines and trained models to disk as binary files to be reused for production purposes.   
  
### What issues does scipy-modeller solve? Why use such a system?  

**Some big problems all Data Science teams need to avoid or solve**  
One major problem data science teams have is that although most data science projects follow the same steps (reading data from a database, preprocessing the data, and building/interpreting a model), every individual scientist/analyst will implement their own functions and workflows (possibly in very different ways) creating messy and hard to manage projects and environments. In other words, it is common that new and growing teams will "reinvent the Data Science wheel" every time a new line project comes along. These tendencies often lead to situations where collaboration is much harder (taking very long to get on the same page) or impossible. Messy, unstructured, worksflows (which are unique to each team member) make collaboration difficult because they add an uneccessary barrier to entry when a new member needs to contribute or take over a project. Instead of only needing to learn  new line of business, the individual also must learn a new workflow entirely (which may be very foreign to them). The process will take a lot of time. If the methods were very well-documented by whoever started/leads the project, the process will take less time. Anyone who has picked up a technical project or a code base knows that the proper documentation often isn't there. This not only increases time to productivity, it also is very dangerous as you cannt validate any procedures without rewriting the methods or workflow yourself. The behaviour of the system may be working as you believe it does, but it also might not. It will take a lot of needless exploration to determine that behaviour. And years later, things still may be failing silently to you.  

In other words, there are many *unknown unknowns*, even when you believe you know how the system works, you can never truly know for certain.  

Why is this a major problem when attempting to build a production model using user data? Because there cannot be ANY unknown unknowns. It is your responsibility to make sure the system is working how you believe it does, how your team believes it does, and how your manager believes it does. It has to be easy to understand within and across teams, and it has to do what you need it do the first time, second time, and the last time its used. And that behaviour must be easy to validate.   

One final problem that must be avoided (or solved after the fact) is allowing dangerous dependecies on one person in terms of the success of a project. Anyone can leave a team or a project at any time. Anyone may need extended time off of work at any time. When only one person is able to run and manage a major project due to their knowledge of their unique workflow and methods, your team will at some point suffer. Someone will leave or take time off and that project will hault for some time. And the painful process of disecting that past, undocumented, unstructured, work will become another employees job (or nightmare). Further bringing down the effectiveness of a team.

**Enabling management and Data Science Leads to be effective**  
The best way to ensure existing and new ensure projects are effectively managed as a team continues to grow is to develop a shared work-flow and consistent environments. To have standards and best practices. To apply established, well-tested, and well-designed methods to new buisness lines is a goal. You do not want your team building new workflows, methods, and data pipelines every single time a new project comes along. That will make a Data Science lead feel uneffective. That will lead to endless amounts of duplicated work and headaches. This library does not only help Data Scientists by providing methods and pipelines they would need to develop anyways, it helps managers and leaders focus on business problems and trust that the technical approach is tested, reliable, and easily extensible.  

**Data Pipelines for Production (and non-production models)**  
Every Data Science team writes pipeline software on top of existing tools. They leverge core or contributed packages from a large community to build on and extend for their own purposes. They need an a consistent way for their team to interact with their backend, extract and format data, preprocess it, model it, validate results, and presist their trained models for future inference in production. They do not write such a pipeline from scratch everytime they need to follow this process. They do it once, they do it well, and they copy and paste that method to each of their lines of business. They write enough code to build a very-well functioning system, and then they focus on refactoring and improving it when needed, and reusing it as much as possible. This library provides just that.

**A Data Processing API**  
Going back to consistency and structure of workflows. Recall that every data scientist must write their own methods to wrangle data. But these methods are always similar for tabular data. Filtering, mutating, expanding the feature space, resampling the data. These are steps we all take. So why would we all implement them in our own way knowing that we are all surely going to make our own mistakes along the way, test our own code, and fix these problems again and again. We should write these methods once, write them well, test them, and use them. This library does just that.
  
  
### Main goals of the library:  
1) greatly simplify and better-structure data preprocessing (including feature extraction/engineering) by exposing an easy to use API which interacts with a database (using the ODBC API via pyodbc) and reads data into/manipulates a Pandas DataFrame object.     
2) implement best-practices for a) working with data in-memory using Pandas (including memory and data type management), b) processing data in a reproducible and intuitive way (structured naming conventions, consistent metholodies, etc.), and c) handling resampling and validation techniques before model results are interpreted.    
3) automate the building of Machine-Learning transformation pipelines and apply their appilication before modelling.    
4) provide a dynamic manager (used as a decorator) to allow the user to construct their own easy to alter and extend high-level worrkflows  however is best for them for each project (capturing the major steps they take).   
5) allow a model to easily be productionalized (including automatic execution of preprocessing streams, transformations, data splitting, candidate model training and validation (with hyperparameter tuning optional), performance evaluation, and persistence of a 'best-fitting' model to a binary file for use for automatic inference.   


### An overview of the process  

The image below shows how scipy-modeller captures each stage of a data science production cycle from data inpute, proprocessing, and transformation, to data splitting, modelling, validation, and persistence. 

![](.rmpics/2020-01-09-13-15-30.png)
<br/>  

As seen above in the transform data step, the "Column transformer pipeline" step requires a better visualization to capture the process. This process can be seen below.   
<br/>  

![](.rmpics/2020-01-09-13-18-44.png)  

This transformation process builds small transformers based on datatypes it discovers in a passed table (which has already been preprocessed with the preprocessing API) and builds those transformers into a larger transformation pipeline.  


### Notes  
--- 
scipy-modeler does a lot of the heavy-lifting of data processing upon reading in from the database. This relies on the use of a 'table_struc', or table structure, passed as a dict to the tables schema. This is user-defined ahead of time (before reading data into memory). The key,value pairs of this dict are to be specified as follows {'column_name': datatype}. For instance: my_table_struc = {'employee_name': Object, 'salary': float, 'seniority_level': category}. ONLY columns you wish to read in from the database should be included, the non-specified columns will never be read into data (an SQL-statement is generated behind the scenes), rather than everything being read into memory and then filtered out later. If no structure is passed, the table schema will default to None, all columns will be read in, and datatypes will be inferrered. Passing a struc dict will often times greatly reduce the memory footprint of the data (especially if the data contains many categorical types) and structure data types, grouping them together, making transformation piplines easier to generate and configure. Passing a struc dict is ALWAYS recommended for each table being read into memory.  

See Jupyter Notebook exampled to understand how these strucs are used.  

Using these strucs is not only clever and efficient (saves many steps one might need to repeat every time she/he reads data into a local dataframe), they greatly reduce the memory footprint of your data.  Below are some example tests showing the memory savings using this approach versus a standard 'read-in, filter and format after' approach.  

![](.rmpics/2020-01-15-10-48-51.png)



