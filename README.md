# AutoDraft
AutoDraft is, primarily, a Streamlit app for visualizing the predictive capabilities of three different models, and their future predictions, when applied to professional hockey players. The motivation for this project is the relative lack of game-level predictions available to fans, in the hopes of being able to identify players that will perform well over the course of a season, as well as when they should be played within it.

This repo therefore includes the app itself, as well as a number of modules that are actively used by the app, or that were used to realize this project's full pipeline. All modules are available in `./AutoDraft/autodraft/`, and can be simply imported as `autodraft.{module}` from within the working folder (`AutoDraft`).

## Setup
First, please clone this repo, and navigate into it:
```
git clone https://github.com/j4th/AutoDraft.git
cd AutoDraft
```

In order to get the app up and running, create a new python virtual environment; for the purpose of this guide, conda will be used. 

If you do not use virtual environemnts, feel free to skip the following step; however, I hope you consider reading up on virtual environments (a good write-up on them is available [here](https://medium.com/@dakota.lillie/an-introduction-to-virtual-environments-in-python-ce16cda92853)).
```
conda create -n autodraft python=3
conda activate autodraft
```
Now, install the app/repo requirements by running:
```
pip install -r requirements.txt
```

## Running the App
This app heavily leverages [Streamlit](https://streamlit.io/) for the front-end, so it is what will be used to run the app itself. Simply do the following to navigate into the working folder, and launch the app:
```
cd AutoDraft
streamlit run app.py
```  
Streamlit will display the address and port the app is being served on, so please simply click the hyperlink. You will be brought to the app in your browser.

## Additional Components
As mentioned earlier, there are some modules that are included within this repo, namely:
 - `api.py`: Wrapper for the NHL's undocumented (mostly, shout-out to [dword4](https://gitlab.com/dword4/nhlapi)) API.
 - `arima.py`: Module for performing Auto-ARIMA modelling and predictions using [pmdarima](https://www.alkaline-ml.com/pmdarima/).
 - `gluonts.py`: Module for performing DeepAR modelling and predictions; could be extended to any [GluonTS](https://gluon-ts.mxnet.io/) model.
 - `visualization.py`: Module for visualizing model output and performance, largely through leveraging [Bokeh](https://bokeh.pydata.org/en/latest/index.html).
All of these can be found in `AutoDraft/autodraft/`, and can be imported using `import autodraft` within the working folder.

## Included Data