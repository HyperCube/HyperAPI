.. hyper_api documentation master file, created by
   sphinx-quickstart on Wed May 16 11:22:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========
Automate your HyperCube workflows thanks to HyperAPI. 

HyperAPI package is already installed in the Jupyter instance accessible from the HyperCube user interface. No external modules are required for the main features.
It is also possible to install the HyperAPI module into an external python 3.6.2 environment. 

Installation in an external python environment
----------------------------------------------
* Go to git project on https://github.com/bearingpoint/bbs-hypercube-api.
* fork the git repo
* clone the fork locally

In a shell, from your source folder, check the HyperAPI version:

	ls dist

Then, install required packages:

On windows::

	pip install –r requirements.txt
	pip install dist/HyperAPI-_x_-py3-none-any.whl

On linux/mac::

	pip3 install –r requirements.txt
	pip3 install dist/HyperAPI-_x_-py3-none-any.whl

nb: Replace '_x_' by the correct version

Authentication
--------------
* Copy the API token from your Settings in HyperCube
* From a Notebook inside HyperCube, paste the token to initialize the Api::

	API_TOKEN = 'PASTE YOUR TOKEN HERE'
	from HyperAPI import Api
	api = Api(token=API_TOKEN)

* From a Notebook outside HyperCube (e.g.: Jupyter Notebook)::

	import sys
	sys.path.insert(0, LOCAL_PATH_TO_HYPERAPI)
	API_TOKEN = 'PASTE YOUR TOKEN HERE'
	from HyperAPI import Api
	H3_URL = 'https://h34a.hcube.io' # or the url of another HyperCube platform / setup
	api = Api(token=API_TOKEN, url=H3_URL)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Examples
==================

Each of these examples is built upon the previous ones so they should be executed in order.

Project
-------
* Create a project: :func:`get_or_create <project.ProjectFactory.get_or_create>`, :func:`create <project.ProjectFactory.create>` 
* List and filter projects: :func:`filter <project.ProjectFactory.filter>`
* Retrieve a project: :func:`get <project.ProjectFactory.get>`
* Set a project as default: :func:`set_as_default <project.Project.set_as_default>`
* Delete a project: :func:`delete <project.Project.delete>` ::

	# Create a project
	PRJ_NAME = "Demo_API_Project"
	project = api.Project.get_or_create(PRJ_NAME)
	
	# List
	print('List of projects:')
	for _project in api.Project.filter(): print('\t- {}'.format(_project.name))
	
	# Retrieve a project
	project = api.Project.get(PRJ_NAME)
	
	# Set project as default
	project.set_as_default()
	
	# Display info about the project
	print(project)
	print('Is "{}" default project ?: {}'.format(project.name, project.is_default))


Dataset
-------

* Create a dataset: :func:`get_or_create <dataset.DatasetFactory.get_or_create>`, :func:`create <dataset.DatasetFactory.create>`, :func:`create_from_dataframe <dataset.DatasetFactory.create_from_dataframe>`
* List and filter datasets: :func:`filter <dataset.DatasetFactory.filter>`
* Retrieve a dataset: :func:`get <dataset.DatasetFactory.get>`
* Set a dataset as default: :func:`set_as_default <dataset.Dataset.set_as_default>`
* Split the dataset: split :func:`split <dataset.Dataset.split>`
* Delete a dataset: :func:`delete <dataset.Dataset.delete>` ::

	# Upload data file on HyperCube platform:
	# In the Notebook interface, click 'Upload Files' button and select the data file
	# This is 'Titanic.csv' in our example
	# The root path where files are stored on the platform is '/mnt/notebookfs'
	
	# Define Dataset Name and Path
	DS_NAME = "Demo_API_Dataset"
	import os
	# on platform:
	DS_PATH = os.path.join(os.getcwd(), 'Titanic.csv')
	# locally:
	# DS_PATH = os.path.join(r'LOCAL FILE PATH', 'LOCAL FILE NAME')
	
	# Create
	dataset = project.Dataset.get_or_create(DS_NAME, DS_PATH, delimiter=';')
	
	# List all datasets belonging to the project
	print(project.Dataset.filter())
	#print(project.datasets)
	
	# Retrieve a dataset
	dataset = project.Dataset.get(DS_NAME)
	
	# Set dataset as default
	dataset.set_as_default()
	
	# Display info about the dataset
	print(dataset)
	print('Is "{}" default dataset ?: {}'.format(dataset.name, dataset.is_default))
	
	# Split dataset : train/test
	dataset_train, dataset_test = dataset.split()
	print('Name of the train dataset: {}'.format(dataset_train.name))
	print('Name of the test dataset: {}'.format(dataset_test.name))


Variable
----------

* List and filter variables: :func:`filter <variable.VariableFactory.filter>`
* Retrieve a variable: :func:`get <variable.VariableFactory.get>`
* Ignore / Keep a variable: :func:`ignore <variable.VariableFactory.ignore>` / :func:`keep <variable.VariableFactory.keep>` ::

	# List variables
	print(dataset.Variable.filter())
	# print(dataset.variables)
	
	# Retrieve a variable
	variable = dataset.Variable.get('Survival_Status')
	variable_2 = dataset.Variable.get('ID')
	
	# Display info about the variable
	print(variable)
	print('Variable missing value count: {}'.format(variable.missing_value_count))
	print('Variable modalities: {}'.format(variable.modalities))
	
	# Ignore a variable
	variable_2.ignore()


Target
-------
* Create a target: :func:`create <target.TargetFactory.create>`, :func:`create_targets <target.TargetFactory.create_targets>`, :func:`create_description <target.TargetFactory.create_description>`
* List and filter targets: :func:`filter <target.TargetFactory.filter>`
* Retrieve a target: :func:`get <target.TargetFactory.get>`
* Delete a target: :func:`delete <target.Target.delete>` ::

	# Create a target or a description
	# Note that the creation of a target automatically creates the corresponding description
	target = project.Target.create(variable, 'alive')
	target_2 = project.Target.create(dataset.Variable.get('Age'), scoreTypes=["Average value", 
	                                 "Standard deviation", "Shift"])
	
	# List targets
	print(project.Target.filter())
	# print(project.targets)
	
	# Get a target or description
	TARGET_NAME = variable.name + '_description'
	description = project.Target.get(TARGET_NAME)
	
	# Display info about the target
	print(target)
	print('Target type: {}'.format(target.indicator_type))


Xray
------

* Create a Xray: :func:`get_or_create <xray.XrayFactory.get_or_create>`, :func:`create <xray.XrayFactory.create>`
* List and filter Xrays: :func:`filter <xray.XrayFactory.filter>`
* Retrieve a Xray: :func:`get <xray.XrayFactory.get>`
* Delete a Xray: :func:`delete <xray.Xray.delete>`

* List Xray variables: :func:`Variable.filter <xrayvariable.XRayVariableFactory.filter>`
* Retrieve a Xray variable: :func:`Variable.get <xrayvariable.XRayVariableFactory.get>`
* Sort Xray variables by contrast rate: :func:`Variable.sort <xrayvariable.XRayVariableFactory.sort>`
* Get contrast rates on a Xray variable: :func:`Variable.contrast_rates <xrayvariable.XRayVariable.contrast_rates>` ::

	# Create Xray
	XRAY_NAME = "Demo_API_Xray"
	xray = project.Xray.get_or_create(dataset, XRAY_NAME, description)
	
	# List Xrays
	print(project.Xray.filter())   # lists on the project
	# print(dataset.xrays)          # lists on the dataset
	
	# Get Xray
	xray = project.Xray.get(XRAY_NAME)
	
	# List Xray variables
	print(xray.Variable.filter())   # unsorted
	# print(xray.variables)          # sorted on contrast rate if possible
	
	# Sort Xray variables by contrast rate
	print(xray.Variable.sort(description.variable_name, reverse=True))
	
	# Retrieve a Xray variable
	xray_var = xray.Variable.get('Age')
	
	# Display info about the Xray variable
	print(xray_var)
	print('contrast rate: {}'.format(xray_var.contrast_rates.get(description.variable_name)))


Ruleset
---------

* Create a Ruleset: :func:`get_or_create <ruleset.RulesetFactory.get_or_create>`, :func:`create <ruleset.RulesetFactory.create>`
* List and filter Rulesets: :func:`filter <ruleset.RulesetFactory.filter>`
* Retrieve a Ruleset: :func:`get <ruleset.RulesetFactory.get>`
* Retrieve rules on the Ruleset: :func:`get_rules <ruleset.Ruleset.get_rules>`
* Minmize the Ruleset : :func:`minimize <ruleset.Ruleset.minimize>`
* Create a Model from the Ruleset: :func:`predict <ruleset.Ruleset.predict>`
* Delete a ruleset: :func:`delete <ruleset.Ruleset.delete>` ::

	# Create Ruleset
	RULESET_NAME = "Demo_API_Ruleset"
	ruleset = project.Ruleset.get_or_create(dataset, RULESET_NAME, target)
	
	other_key_indicator_1 = project.Ruleset.create_kpi_option(target_2, 
	                                                          average_value_min=30, 
	                                                          average_value_max=100)
	RULESET_NAME_2 = "Demo_API_Ruleset_2"
	ruleset_2 = project.Ruleset.get_or_create(dataset, RULESET_NAME_2, target, 
	                                          compute_other_key_indicators=[other_key_indicator_1])
	
	# List Rulesets
	print(project.Ruleset.filter())   # lists on the project
	# print(dataset.rulesets)          # lists on the dataset
	
	# Retrieve a Ruleset
	ruleset = project.Ruleset.get(RULESET_NAME)
	
	# Retrieve all rules on the Ruleset
	ruleset.get_rules()
	
	# Minimize the ruleset
	minimized_ruleset = ruleset.minimize('Demo_API_Minimized_Ruleset')
	
	# Create a Model from the Ruleset
	model = ruleset.predict(dataset, 'Demo_API_Model_from_Ruleset', target)
	print(model)


Model
-------
* Create a HyperCube model: :func:`get_or_create_hypercube <model.ModelFactory.get_or_create_hypercube>`, :func:`create_hypercube <model.ModelFactory.create_hypercube>`
* List and filter models: :func:`filter <model.ModelFactory.filter>`
* Retrieve a model: :func:`get <model.ModelFactory.get>`
* Delete a model: :func:`delete <model.Model.delete>`

* Score the model on a test set: :func:`predict_scores <model.ClassifierModel.predict_scores>`
* Apply the model (As in the HyperCube UI): :func:`apply <model.ClassifierModel.apply>`
* Export the scores: :func:`export_scores <model.ClassifierModel.export_scores>`
* Export the model: :func:`export_model <model.HyperCube.export_model>`

* Display curves: :func:`display_curve <model.ClassifierModel.display_curve>`
* Confusion matrix: get_confusion_matrix: :func:`get_confusion_matrix <model.ClassifierModel.get_confusion_matrix>` ::

	# Create a HyperCube model
	MODEL_NAME = "Demo_API_Model"
	model = project.Model.get_or_create_hypercube(dataset=dataset_train, name=MODEL_NAME, target=target)
	
	# List Models
	print(project.Model.filter())
	#print(project.models)
	
	# Retrieve a Model
	model = project.Model.get(MODEL_NAME)
	
	# Display info about a Model
	print(model)
	print('Model name : {}'.format(model.name))

	# Score the test dataset
	scores = model.predict_scores(dataset_test)
	print('type of output: {}'.format(type(scores)))
	print(scores)
	
	# Apply the model as in the HyperCube UI
	applied_model = model.apply(dataset_test, 'Demo_API_Applied_Model')
	print(applied_model)

	# Export the scores
	SCORE_FILE=r'ENTER PATH HERE\scores.csv'
	applied_model.export_scores(SCORE_FILE)

	# Export the model
	MODEL_FILE=r'ENTER PATH HERE\H3Model.py'
	applied_model.export_model(MODEL_FILE, format='Python')
	
	# Display curves
	model.display_curve()                        # ROC curve
	model.display_curve(curve = 'Lift curve')
	model.display_curve(curve = 'Gain curve')
	model.display_curve(curve = 'Precision Recall')

	# Confusion matrix
	conf = model.get_confusion_matrix(0.1)
	conf.false_negatives

API reference
=====================================   

Project
-------

.. automodule:: project
   :members:
   :member-order: bysource

Dataset
-------

.. automodule:: dataset
   :members:
   :member-order: bysource

Variable
--------

.. automodule:: variable
   :members:
   :member-order: bysource

Target
------

.. automodule:: target
   :members:
   :member-order: bysource

Xray
----

.. automodule:: xray
   :members:
   :member-order: bysource

.. automodule:: xrayvariable
   :members:
   :member-order: bysource
   
Ruleset
-------

.. automodule:: ruleset
   :members:
   :member-order: bysource
   
Model
-----

.. automodule:: model
   :members:
   :member-order: bysource

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
