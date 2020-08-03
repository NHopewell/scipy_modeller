The primary contact for this for this project is Nick Hopewell  
<nicholashopewell@gmail.com>    


# scipy-modeller  
--- 
An overview of some applications and goals of scipy-modeller
---
The scipy-modeller library is a Data Science library which supports data extractiong, preprocessing, and predictive modelling by providing well-tested methods for common tasks which every data scientist must perform. This library construct, and persist, transfomer pipelines used during predictive modelling. A key component of this project is a data preprocessing API useful when wrangling data prior to modelling. By providing a set of well-tested reuseable methods for common taks, a lot of redundancy and replication of efforts are avoided.
  
  
### Main goals of the library:  
1) greatly simplify and better-structure data preprocessing by exposing an easy to use API which interacts with a database (using the ODBC API via pyodbc) and reads data into/manipulates a Pandas DataFrame object.     
2) implement best-practices for a) working with data in-memory using Pandas (including memory and data type management), and b) processing data in a reproducible and intuitive way (structured naming conventions, consistent methodologies, etc.).
3) automate the building of Machine-Learning transformation pipelines and apply them automatically before modelling.    
4) allow a model to easily be productionalized. This includes execution of preprocessing streams, transformations, data splitting, candidate model training and validation, performance evaluation, and persistence of a 'best-fitting' model to disk for automatic inference later on.   
5) To have assumptions programmed as assertions to define what user behaviour is expected at each stage in the process. 


### An overview of the process  

The image below shows how scipy-modeller captures each stage of a data science production cycle from data read-in, preprocessing, and transformation, to data splitting, modelling, validation, and persistence. 

![](rmpics/2020-01-09-13-15-30-v2.PNG)
<br/>  

As seen above in the transform data step, the "Column transformer pipeline" step requires a better visualization to capture the process. This process can be seen below.   
<br/>  

![](rmpics/2020-01-09-13-18-44.png)  

This transformation process builds small transformers based on datatypes it discovers in a passed table (which has already been preprocessed with the preprocessing API) and builds those transformers into a larger transformation pipeline.  


### Notes  
--- 
scipy-modeler does a lot of the heavy-lifting of data processing upon reading in from the database. This relies on the use of a 'table_struc', or table structure, passed as a dict to the tables schema. This is user-defined ahead of time (before reading data into memory). The key,value pairs of this dict are to be specified as follows {'column_name': datatype}. For instance: my_table_struc = {'employee_name': Object, 'salary': float, 'seniority_level': category}. ONLY columns you wish to read in from the database should be included, the non-specified columns will never be read into data (an SQL-statement is generated behind the scenes), rather than everything being read into memory and then filtered out later. If no structure is passed, the table schema will default to None, all columns will be read in, and datatypes will be inferred. Passing a struc dict will often times greatly reduce the memory footprint of the data (especially if the data contains many categorical types) and structure data types, grouping them together, making transformation piplines easier to generate and configure. Passing a struc dict is ALWAYS recommended for each table being read into memory.

Using these strucs not only prevents a lot of headaches when dealing with data types (saves many steps one might need to repeat every time she/he reads data into a local dataframe), they also greatly reduce the memory footprint of your data. Below are some example tests showing the memory savings using this approach versus a standard 'read-in, filter and format after' approach.
  

![](rmpics/2020-01-15-10-48-51.png)



