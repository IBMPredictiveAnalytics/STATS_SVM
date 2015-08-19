#/***********************************************************************
# * Licensed Materials - Property of IBM 
# *
# * IBM SPSS Products: Statistics Common
# *
# * (C) Copyright IBM Corp. 2014
# *
# * US Government Users Restricted Rights - Use, duplication or disclosure
# * restricted by GSA ADP Schedule Contract with IBM Corp. 
# ************************************************************************/

# author__ = "SPSS, JKP"
# version__ = "1.0.1"

# History
# 12-May-2014 Original Version


helptext="STATS SVM 
    CMDMODE=estimate* or predict
    DEPENDENT=dependent variable INDEP=independent variables ID=id variable
    SVMTYPE=AUTOMATIC* or CCLASSIFICATION or NUCLASSIFICATION or
    ONECLASSIFICATION or EPSREGRESSION or NUREGRESSION
    KERNEL=LINEAR or POLYNOMIAL or RADIAL* or SIGMOID
    WORKSPACEFILE='R workspace file containing the model for predict mode'
/OPTIONS
    SCALE=YES* or NO
    *** the follow parameters apply as appropriate and can be single
    *** values or a list of values to use in a grid search for the best model,
    which may be time consuming.
    DEGREE=3 GAMMA=1/(number of independent variables) COEF0=0 COST=1 EPSILON=.1
    NU=.5

    CLASSWEIGHTS = dependent variable value weight ...
    NUMCROSSFOLDS = number of cross folds in k-fold validation
    PROBPRED = YES or NO* for class probability predictions
    MISSING = OMIT* or FAIL
/SAVE 
    DATASET=new dataset name
    WORKSPACE = CLEAR* or RETAIN
    WORKSPACEFILE = file specification for saving model
    DECISIONVALUES = YES or NO*
/OUTPUT
    FEATUREWEIGHTS=YES* or NO

STATS SVM /HELP.  prints this information and does nothing else.

Example:
    STATS SVM DEPENDENT=Species INDEP=sepallength sepalwidth ID=obs
    /OPTIONS GAMMA = .1 .2 .3 COST=100
    /SAVE DATASET = predicted WORKSPACEFILE='C:/mymodels/iris.Rdata'.

    STATS SVM CMDMODE=PREDICT WORKSPACEFILE='c:/mymodels/iris.Rdata'
    /SAVE DATASET = newpredicted.

This command operates in two modes.  If CMDMODE=ESTIMATE, the model is built,
and, optionally, the fitted values for the dependent variable are saved to a new
dataset.  If CMDMODE=PREDICT, a previously estimated model is loaded from a file
or is taken from the current workspace, and a new dataset with predicted values
is generated.  All estimation parameters are supplied only in estimate mode.  
In predict mode, no model parameters are specified, but the model source 
is required.  Any estimation parameters specified are ignored, but the ID
variable can be different from the estimation ID.
The model independent variables are read from the Statistics active dataset.

For estimate, all the cases are considered the training dataset.  Select
holdout cases before calling the procedure, and use predict mode to
assess performance.

If the dependent variable is categorical, the SVM is a classification model;
otherwise it is a regression model, except that if there is no dependent
variable, it can be used for detecting anomalies.  For classification
models with more than two classes, a model is estimated for each pair
of values and predictions are made by voting.

Categorical predictors 
are automatically converted to factors.  The SVM types are for classification
or regression as their names indicate, defaulting to CCLASSIFICATION or 
EPSREGRESSION.  If no dependent variable is specified, ONECLASSIFICATION 
is assumed.

There are four kernel types with various parameters.
LINEAR     u' * v
POLYNOMIAL (GAMMA * u' * v + COEF0)**DEGREE)
RADIAL     exp(-GAMMA * |u-v|**2)
SIGMOID     tanh(GAMMA * u' * v + COEF0)
where u and v are independent variable cases.

NU is used in the NU- or ONE- SVM types.  0 <= nu < 1.

The kernel parameters as well as NU and EPSILON can be specified 
as single values or as lists of values.  If any parameter has 
more than one value, a grid search is carried out to find the 
best values for the model.  The specified or selected values 
are displayed in an output table.

For prediction, the source of the already estimated model is
specified in the WORKSPACEFILE parameter of the first part of the 
command.  It can be a quoted file specification or * to indicate that
the model is in the current workspace as might be the case if the
model was just estimated, and the workspace was retained.

SCALE indicates whether the data should be scaled or not.  If YES,
a table of the scaling factors is displayed.  When predicting,
the scaling transform from the estimation data is applied to the new
input data.

CLASSWEIGHTS can specify nonuniform weights for the factor levels.
of a classification model.  It is written as pairs of factor 
values and weights.  For example,
'clerical' .5 'custodial' 1 'manager' 2.  
The levels should be string or numeric corresponding to the 
type of the dependent variable.  Omitted classes get a weight of 1.

MISSING=OMIT causes cases with any missing values to be omitted.
MISSING=FAIL causes the procedure to stop if there are any missing values.

NUMCROSSFOLDS specifies k-fold cross validation.  If k > 0, a table
of accuracies is displayed.

For both estimate and predict modes, the predicted values can be
written to a new dataset specified by DATASET.  The dataset name
must not already be in use.  If an ID variable was specified,
it is included in the dataset.  Since missing value deletion
may cause some cases not to appear in the dataset, an ID may
be necessary in to match the dataset back to the input.

If PROBPRED is YES at estimation time and it is classification
model, when a dataset is created in predict mode, the probability
for each class is included in the output dataset.

The SAVE subcommand WORKSPACEFILE parameter, which applies only
in estimate mode, specifies that the model be written to the 
specified file.  

DECISIONVALUES specifies whether or not to include these in the 
prediction dataset.  They are not provided if the dependent 
variable has only two values.  If a categorical
dependent variable has k values, there will be k(k-1)/2
additional variables in the dataset.

If WORKSPACE = CLEAR, the model is deleted
from memory while RETAIN causes it to be available for use
in predict mode in the current session unless the workspace
has been otherwise cleared.  Multiple uses of predict
mode can use the retained workspace as long as
WORKSPACE=RETAIN is used each time.

FEATUREWEIGHTS specifies whether or not to display a table
of variable weights for the model.

"
svmtypemap = list(cclassification="C-classification", nuclassification="nu-classification",
    oneclassification="one-classification", epsregression="eps-regression",
    nuregression="nu-regression")

### MAIN ROUTINE ###
dosvm = function(cmdmode="estimate", dependent=NULL, indep=NULL, 
            id=NULL, svmtype="automatic", featureweights=FALSE, epsilon=.1,
            scale=TRUE, kernel="radial", workspace=NULL, degree=3, gamma=NULL,
            coef0=0, cost=1, nu=.5, classweights=NULL, numcrossfolds=0,
            probpred=FALSE, missingvalues="omit", dataset=NULL, plotit=FALSE,
            workspaceaction="clear", workspaceoutfile=NULL, decisionvalues=FALSE) {
    # Estimate or predict a support vector machine
    
    setuplocalization("STATS_SVM")
    
    # A warnings proc name is associated with the regular output
    # (and the same omsid), because warnings/errors may appear in
    # a separate procedure block following the regular output
    procname=gtxt("SVM")
    warningsprocname = gtxt("SVM: Warnings")
    omsid="STATSSVM"
    warns = Warn(procname=warningsprocname,omsid=omsid)

    tryCatch(library(e1071), error=function(e){
        warns$warn(gtxtf("The R %s package is required but could not be loaded.", "e1071"),dostop=TRUE)
        }
    )
    predmode = cmdmode == "predict"
    if (!is.null(workspaceoutfile)) {
        workspaceoutfile = gsub("[\\]", "/", workspaceoutfile)
    }
    if (!is.null(workspace)) {
        workspace = gsub("[\\]", "/", workspace)
    }
    if (predmode) {
        if (is.null(dataset)) {
            warns$warn(gtxt("Prediction requires a dataset specification for output"),
            dostop=TRUE)
        }
        if (is.null(workspace)) {
            warns$warn(gtxt("A workspace specification is required for prediction"), dostop=TRUE)
        }
        if (workspace !="*") {
            tryCatch(load(workspace), 
            error=function(e) {warns$warn(gtxt("Workspace file could not be loaded"),
                dostop=TRUE)
            }
            )
        }

        if (!exists("svmres") || !("svm" %in% class(svmres))) {
            warns$warn(gtxt("No SVM model was found in the specified workspace"), dostop=TRUE)
        }
        alldata = allargsp[["indep"]]
        dependent = NULL
    } else {
        if (is.null(indep)) {
            warns$warn(gtxt("No independent variables were specified.  At least one is required"),
                dostop=TRUE)
        }
    alldata=c(indep, dependent)
    }
    if (!is.null(dataset)) {
        alldatasets = spssdata.GetDataSetList()
        if ("*" %in% alldatasets) {
            warns$warn(gtxt("The active dataset must have a name in order to use this procedure"),
                dostop=TRUE)
        }
        if (dataset %in% alldatasets) {
            warns$warn(gtxt("The output dataset name must not already be in use"),
                dostop=TRUE)
        }
    }
    dta = tryCatch(spssdata.GetDataFromSPSS(alldata, row.label=id, missingValueToNA=TRUE,
        factorMode="levels"),
        error=function(e) {warns$warn(e$message, dostop=TRUE)}
        )
    if (!predmode) {
        isfactor = sapply(dta, is.factor) # for checking levels when predicting
        if (is.null(dependent)) {
            svmtype = "oneclassification"
            dofitted=FALSE

        } else {
            isfactor = isfactor[-length(isfactor)]  # we don't care about the dep variable
            if (is.null(svmtype) || svmtype == "automatic") {
                svmtype = ifelse(is.factor(dta[[dependent]]), "cclassification", "epsregression")
            }
            dofitted = is.factor(dta[[dependent]])
        }
        thesvmtype = svmtypemap[svmtype]
        if (is.null(kernel)) {
            kernel="radial"
        }
        if (kernel == "automatic") {
            kernel="radial"
        }
        if (is.null(gamma) && !is.null(indep)) {
            gamma= 1/length(indep)
        }
        if (is.null(dependent)) {
            frml = as.formula(paste("~", paste(indep, collapse="+"), sep=""))
        } else {
            frml = as.formula(paste(dependent, "~", paste(indep, collapse="+"), sep=""))
        }
        if (missingvalues == "omit") {
            naaction = na.omit
        } else {
            naaction = na.fail
        }
        if (!is.null(classweights)) {
            # convert list to named list
            cwlen = length(classweights)
            if (cwlen %% 2 != 0) {
                warns$warn(gtxt("Incorrect class weights: weights must be specified as pairs of values and weights"),
                    dostop=TRUE)
            }
            cwnames = classweights[seq(1, cwlen,2)]
            classweights = classweights[seq(2, cwlen,2)]
            names(classweights) = cwnames
        }
        modeldate = date()
    } else { # prediction mode
        isfactorp = sapply(dta, is.factor)
        if (!identical(allargsp[["isfactor"]], isfactorp)) {
            warns$warn(gtxt("Prediction variables measurement levels do not match estimation variables"),
            dostop=TRUE)
        }
    }
    
    allargs = as.list(environment())
    allargs[["dta"]] = NULL

    # The prediction-mode dataset is constructed by the savepred function
    performances = NULL
    if (cmdmode == "estimate") {
        if (is.null(gamma)) {
            gamma = 1/length(indep)
        }
        args = list(formula=frml, data=dta, scale=scale, type=thesvmtype,
                    kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, epsilon=epsilon,
                    cost=cost, nu=nu, class.weights=classweights, cross=numcrossfolds,
                    probability=probpred, na.action=naaction,
                    fitted=dofitted || !is.null(dataset))
        # update inputs with the optimal values if ranges were given
        tunedparms = tuner(dependent, frml, dta, kernel, degree, gamma, coef0, epsilon, cost, nu, 
            classweights, naaction=naaction, warns)
        performances = tunedparms[["performances"]]  # may be NULL
        tunedparms$performances = NULL
        args[names(tunedparms)] = tunedparms

        args = args[!sapply(args, is.null)]  # remove all null arguments
        svmres = tryCatch(do.call(svm, args),
            error = function(e) {warns$warn(e$message, dostop=TRUE)}
        )
    }
    displayresults(allargs, allargsp, svmres, tunedparms, dta, performances, warns)
    if (!is.null(dataset)) {
        savepred(allargs, allargsp, svmres, dta, warns)
    }
    if (!is.null(workspaceoutfile)) {
        if (cmdmode == "predict") {
            warns$warn(gtxt("Ignoring workspace save request because using predict mode"),
                dostop=FALSE)
        } else {
            allargsp = allargs
            svmres$fitted = NULL
            save(svmres, allargsp, tunedparms, file=workspaceoutfile)
        }
    }
    if (workspaceaction == "clear") {
        rm(list=ls())
        rm(list=ls(envir=.GlobalEnv), envir=.GlobalEnv)
    } else {
        if (cmdmode == "estimate") {
            assign("svmres", svmres, envir=.GlobalEnv)
            assign("allargsp", allargs, envir=.GlobalEnv)
            assign("tunedparms", tunedparms, envir=.GlobalEnv)
        }
    }
}

tuner = function(dependent, frml, dta, kernel, degree, gamma, coef0, epsilon, cost, nu, 
            classweights, naaction, warns) {
    # apply tuning if any parameters are multiples
    
    parms = list(degree=degree, gamma=gamma, coef0=coef0, 
        epsilon=epsilon, cost=cost, nu=nu)
    if (max(sapply(parms, length)) == 1) {
        return(NULL)   # no ranges were given
    } else {
        if (is.null(dependent)) {
            warns$warn(gtxt("Tuning cannot be used if there is no dependent variable"),
                dostop=TRUE)
        }
    }
    b = tune(svm, train.x=frml, data=dta, kernel=kernel, ranges=parms, 
        class.weights=classweights, na.action=naaction)
    parms[names(b$best.parameters)] = b$best.parameters
    parms[["performances"]] = b$performances
    return(parms)
}

displayresults = function(allargs, allargsp, svmres, tunedparms, dta, performances, warns) {
    # display results
    # allargs is the parameter set (estimation or prediction)
    # allargsp is the saved model specifications if predicting
    # svmres is the estimated model
    # tunedparms is the actual parameters used after any tuning
    # performances is a data frame of performance statistics and can be NULL
    
    StartProcedure(allargs[["procname"]], allargs[["omsid"]])
    
    # summary results
    # input specifications
    if (allargs[["cmdmode"]] == "estimate") {
        asrc = allargs
    } else {
        asrc = allargsp
        asrc["workspaceoutfile"] = NULL   # save only valid for estimation
    }
    lbls = c(gtxt("Dependent Variable"),
             gtxt("Independent Variables"),
             gtxt("SVM Type"),
             gtxt("Kernel"),
             gtxt("Model Source"),
             gtxt("Scale Inputs"),
             gtxt("Kernel Degree"),
             gtxt("Kernel Gamma"),
             gtxt("Kernel Coef0"),
             gtxt("Misclassification Cost"),
             gtxt("Nu"),
             gtxt("Epsilon"),
             gtxt("Class Weights"),
             gtxt("Number of Crossfolds"),
             gtxt("Output Dataset"),
             gtxt("Result Workspace"),
             gtxt("Workspace Contents"),
             gtxt("Model Estimation Date")

    )
    # TODO: a little more translation work
    vals = c(
            ifelse(is.null(asrc[["dependent"]]), gtxt("<None>"), asrc["dependent"]),
            paste(asrc[["indep"]], collapse=", "),
            asrc[["svmtype"]],
            asrc[["kernel"]],
            ifelse(is.null(allargs[["workspace"]]), gtxt("New"), 
                paste('"', allargs[["workspace"]], '"', sep="")),
            ifelse(asrc[["scale"]], gtxt("Yes"), gtxt("No")),
            ifelse(asrc[["kernel"]] %in% list("linear", "radial", "sigmoid"), 
                gtxt("NA"), svmres[["degree"]]),
            ifelse(asrc[["kernel"]] == "linear", 
                gtxt("NA"), svmres[["gamma"]]),
            ifelse(asrc[["kernel"]] %in% list("linear", "radial"), 
                gtxt("NA"), svmres[["coef0"]]),
#             ifelse(is.null(svmres[["degree"]]), gtxt("NA"), svmres[["degree"]]),
#             ifelse(is.null(svmres[["gamma"]]), gtxt("NA"), svmres[["gamma"]]),
#             ifelse(is.null(svmres[["coef0"]]), gtxt("NA"), svmres[["coef0"]]),
            ifelse(is.null(asrc[["cost"]]), gtxt("NA"), gtxt(asrc[["cost"]])),
            ifelse(is.null(svmres[["nu"]]), gtxt("NA"), svmres[["nu"]]),
            ifelse(is.null(svmres[["epsilon"]]), gtxt("NA"), svmres[["epsilon"]]),
            ifelse(is.null(asrc[["classweights"]]), gtxt("None"), 
                paste(names(asrc[["classweights"]]), asrc[["classweights"]], collapse=", ")),
            asrc[["numcrossfolds"]],
            ifelse(is.null(allargs[["dataset"]]), gtxt("NA"), allargs[["dataset"]]),
            ifelse(is.null(asrc[["workspaceoutfile"]]), gtxt("NA"), asrc[["workspaceoutfile"]]),
            ifelse(asrc[["workspaceaction"]] == "clear", gtxt("Clear"), gtxt("Retain")),
            asrc[["modeldate"]]
    )

    # settings and result summary
    if (!is.null(tunedparms)) {
        caption=gtxt("Some parameters were tuned from a list of choices")
    } else {
        caption=""
    }
    spsspivottable.Display(data.frame(cbind(vals), row.names=lbls), title = gtxt("Summary"),
                           collabels=c(gtxt("Summary")), templateName="SVMSUMMARY", outline=gtxt("Summary"),
                           caption = paste(caption, 
                                gtxt("Computations done by R package e1071"), sep="\n")
    )
    if (!is.null(performances)) {
        names(performances)[[(ncol(performances)-1)]] = gtxt("Error")
        names(performances)[[ncol(performances)]] = gtxt("Dispersion")
        spsspivottable.Display(performances, title=gtxt("Parameter Tuning Performance"),
            templateName="SVMTUNINGPERF", outline=gtxt("Tuning Performance"),
            caption=paste(gtxt("Error is classification error or mean squared error depending on dependent variable type."),
                gtxt("Parameters actually used depend on kernel choice and svm type."), sep="\n")
        )
    }
    # Scaling
    if (allargs[["cmdmode"]] == "estimate" && asrc[["scale"]]) {
        df = data.frame(svmres[["x.scale"]])
        if (!is.null(svmres[["y.scale"]])) {
            ys = data.frame(svmres[["y.scale"]])
            df = rbind(df, ys)
            row.names(df)[nrow(df)] = allargs[["dependent"]]
        }
        if (nrow(df) > 0) {
            names(df) = list(gtxt("Center"), gtxt("Scale"))
            spsspivottable.Display(df, title=gtxt("Variable Scaling"), templateName="SVMSCALE",
                outline=gtxt("Scaling", rowdim=gtxt("Variables"), hiderowdimtitle=FALSE)
            )
        }
    }
    # Crossfold validation
    if (allargs[["cmdmode"]] == "estimate" && allargs[["numcrossfolds"]] > 0) {
        if (!is.null(svmres[["accuracies"]])) {
            d =svmres[["accuracies"]]
            d[length(d) + 1] = svmres[["tot.accuracy"]]
            df = data.frame(d)
            names(df) = gtxt("Accuracies")
            row.names(df)[nrow(df)] = gtxt("Total Accuracy")
            spsspivottable.Display(df, title=gtxt("Crossfold Accuracies"),
                templateName="SVMACCURACIES", outline=gtxt("Accuracies")
            )
        } else {
            d = svmres[["MSE"]]
            d[length(d)+1] = svmres[["tot.MSE"]]
            d[length(d)+1] = svmres[["scorrcoeff"]]
            df = data.frame(d)
            row.names(df)[(nrow(df)-1):(nrow(df))] = c(gtxt("Total MSE"), 
                gtxt("Squared Correlation, Predicted and Actual"))
            spsspivottable.Display(df, title=gtxt("Crossfold Mean Squared Errors"),
                templateName="SVMMSES", outline=gtxt("Mean Squared Errors")
            )
        }
    }
    # feature weights
    if (allargs[["cmdmode"]] == "estimate") {
        if (allargs[["featureweights"]]) {
            if (svmres$tot.nSV > 0) {
                df = data.frame(t(t(svmres[["coefs"]]) %*% svmres[["SV"]]))
                names(df) = 1:ncol(df)
                spsspivottable.Display(df, title=gtxt("Feature Weights"), templateName="SVMWEIGHTS",
                    outline=gtxt("Feature Weights"),
                    caption=gtxtf("Number of support vectors: %s", svmres$tot.nSV))
            } else {
                warns$warn(gtxt("Feature weights are not produced, because there are no support vectors"),
                    dostop=FALSE)
            }
        }
        # confusion table (estimate mode only)
        if (!is.null(allargs[["dependent"]]) && is.factor(svmres[["fitted"]])) {
            # merge needed because of possible missing data deletion in fitted values
            tbldta = merge(dta[allargs[["dependent"]]], as.data.frame(svmres[["fitted"]]), by="row.names")
            tbl = table(tbldta[[2]], tbldta[[3]])
            correct = sum(diag(tbl), na.rm=TRUE) / sum(tbl, na.rm=TRUE) * 100
            tbl = as.data.frame.matrix(addmargins(tbl, margin=c(1,2)))
            names(tbl)[[length(names(tbl))]] = gtxt("Total")
            row.names(tbl)[length(row.names(tbl))] = gtxt("Total")
            nelt = nrow(tbl)
            pctrow = vector(length=nelt)
            pctcol = vector(length=nelt)
            for (i in 1:(nelt-1)) {
                pctrow[i] = round(tbl[i,i]/tbl[i, nelt] * 100, digits=2)
                pctcol[i] = round(tbl[i,i]/tbl[nelt, i] * 100, digits=2)
            }
            pctcol[nelt] = NA
            tbl[, nelt+1] = pctrow
            tbl[nelt+1,] = c(pctcol, NA)
            tbl[nelt, nelt+1] = round(correct, digits=2)
            names(tbl)[nelt+1] = gtxt("% Correct")
            row.names(tbl)[nelt+1] = gtxt("% Correct")
            spsspivottable.Display(tbl, title=gtxt("Confusion"), templateName="SVMCONFUSION",
                rowdim=gtxt("Actual"), hiderowdimtitle=FALSE,
                coldim=gtxt("Fitted"), hidecoldimtitle=FALSE,
                format=formatSpec.GeneralStat,
                outline=gtxt("Confusion"),
                caption=gtxtf("Percent Correct: %.3f", correct)
            )
        }
    }

    spsspkg.EndProcedure()
}

savepred = function(allargs, allargsp, svmres, dta, warns) {
    # save predicted values
    # in estimate mode, these come from the fitted values
    # in predict mode, they are constructed here
    
    dict = list()

    depname = ifelse(is.null(allargs[["dependent"]]), "", allargs[["dependent"]])
    dataset = allargs[["dataset"]]  # must have  been specified
    hasprob = FALSE
    hasdvs = FALSE
    if (allargs[["cmdmode"]] == "estimate") {
        idname = ifelse(is.null(allargs[["id"]]), "", allargs[["id"]])  # name of ID variable, if any
        pred = data.frame(svmres[["fitted"]])
        rnlength = max(nchar(row.names(pred))) * 3 #unicode worst-case expansion
    } else { # predicting on new data
        idname = ifelse(is.null(allargsp[["id"]]), "", allargsp[["id"]])
        # check that all predictors are supplied
        predictors = as.character(attr(svmres$terms, "predvars")[-1]) # list includes dependent variable
        if (!is.null(allargsp[["dependent"]])) {
            deploc = match(allargsp[["dependent"]], predictors)
            if (!is.na(deploc)) {
                predictors = predictors[-deploc]
            }
        }
        missingpred = setdiff(predictors, names(dta))
        if (length(missingpred) > 0) {
            warns$warn(gtxtf("The following predictors are missing from the input data: %s",
                paste(missingpred, collapse=", " )), dostop=TRUE)
        }
        pred = tryCatch(predict(svmres, newdata=dta, probability=allargs[["probpred"]],
                    decision.values=allargs[["decisionvalues"]]),
                error=function(e) {warns$warn(e$message, dostop=TRUE)}
                )
        if (allargs[["decisionvalues"]]) {
            # preserve full attribute names to use as value labels
            dvs = data.frame(attr(pred, "decision.values"), check.names=FALSE)
            if (nrow(dvs) > 0) {
                dvlabels = names(dvs)
                names(dvs) = paste("DV", 1:ncol(dvs),sep="")
                hasdvs = TRUE
            }
        }
        if (is.null(attr(pred, "probabilities"))) {
            if (allargsp[["probpred"]]) {
            warns$warn(gtxt("Class probabilities were requested but are not available"),
                dostop=FALSE)
            }
            pred = data.frame(pred)
        } else {
            pred = data.frame(pred, attr(pred, "probabilities"))
            hasprob = TRUE
        }
        
        rnlength = max(nchar(row.names(pred))) * 3
    }
    if (nrow(pred) == 0) {
        warns$warn(gtxt("All cases have missing data.  No predictions are produced"),
            dostop=TRUE)
    }

    if (is.factor(pred[[1]])) {
        if (allargs[["cmdmode"]] == "estimate") {
            dep = allargs[["dependent"]]
        } else {
            dep = allargsp[["dependent"]]
        }
        catdict = spssdictionary.GetCategoricalDictionaryFromSPSS(dep)
        if (!is.null(catdict[["name"]])) {
            catdict[["name"]] = "Predicted"
            # the set api fails if these are numeric
            catdict$dictionary[[1]]$levels = sapply(catdict$dictionary[[1]]$levels, as.character)
        } else {
            catdict = NULL
        }
        pred[[1]] = as.character(pred[[1]])
        prnlength = max(nchar(pred[[1]])) * 3
        format = paste("A", prnlength, sep="")
        mlevel = "nominal"
    } else {
        catdict = NULL
        prnlength = 0
        format="F8.2"
        mlevel = "scale"
    }
    # row names will always be nominal strings
    dict[[1]] = c("ID", idname, rnlength, paste("A", rnlength, sep=""), "nominal")
    dict[[2]] = c("Predicted", depname, prnlength, format, mlevel)
    # additional variables for probabilities if available
    index = 2 # starting pont for decision values if no probs
    if (hasprob) {
        cnames = names(pred)
        for (index in 2:ncol(pred)) {
            dict[[index+1]] = c(paste("Prob", index-1, sep=""), cnames[[index]],
            0, "F6.4", "scale")
        }
        index=index+1
    }
    if (hasdvs) {
        pred = data.frame(pred, dvs)
        dvnames = names(dvs)
        for (i in 1:length(dvlabels)) {
            dict[[index+i]] = c(dvnames[[i]], dvlabels[[i]], 0, "F8.4", "scale")
        }
    }
    dict = spssdictionary.CreateSPSSDictionary(dict)
    if (is.null(catdict)) {  # api won't accept a null category dictionary
        spssdictionary.SetDictionaryToSPSS(dataset, dict)
    } else {
        spssdictionary.SetDictionaryToSPSS(dataset, dict, catdict)
    }
    tryCatch(spssdata.SetDataToSPSS(dataset, data.frame(row.names(pred), pred)),
        error=function(e) {warns$warn(e$message, dostop=TRUE)}
    )
    spssdictionary.EndDataStep()
    
}

Warn = function(procname, omsid) {
    # constructor (sort of) for message management
    lcl = list(
        procname=procname,
        omsid=omsid,
        msglist = list(),  # accumulate messages
        msgnum = 0
    )
    # This line is the key to this approach
    lcl = list2env(lcl) # makes this list into an environment

    lcl$warn = function(msg=NULL, dostop=FALSE, inproc=FALSE) {
        # Accumulate messages and, if dostop or no message, display all
        # messages and end procedure state
        # If dostop, issue a stop.

        if (!is.null(msg)) { # accumulate message
            assign("msgnum", lcl$msgnum + 1, envir=lcl)
            # There seems to be no way to update an object, only replace it
            m = lcl$msglist
            m[[lcl$msgnum]] = msg
            assign("msglist", m, envir=lcl)
        } 

        if (is.null(msg) || dostop) {
            lcl$display(inproc)  # display messages and end procedure state
            if (dostop) {
                stop(gtxt("End of procedure"), call.=FALSE)  # may result in dangling error text
            }
        }
    }
    
    lcl$display = function(inproc=FALSE) {
        # display any accumulated messages as a warnings table or as prints
        # and end procedure state, if any

        if (lcl$msgnum == 0) {   # nothing to display
            if (inproc) {
                spsspkg.EndProcedure()
            }
        } else {
            if (!inproc) {
                procok =tryCatch({
                    StartProcedure(lcl$procname, lcl$omsid)
                    TRUE
                    },
                    error = function(e) {
                        FALSE
                    }
                )
            }
            if (procok) {  # build and display a Warnings table if we can
                table = spss.BasePivotTable("Warnings ","Warnings") # do not translate this
                rowdim = BasePivotTable.Append(table,Dimension.Place.row, 
                    gtxt("Message Number"), hideName = FALSE,hideLabels = FALSE)

                for (i in 1:lcl$msgnum) {
                    rowcategory = spss.CellText.String(as.character(i))
                    BasePivotTable.SetCategories(table,rowdim,rowcategory)
                    BasePivotTable.SetCellValue(table,rowcategory, 
                        spss.CellText.String(lcl$msglist[[i]]))
                }
                spsspkg.EndProcedure()   # implies display
            } else { # can't produce a table
                for (i in 1:lcl$msgnum) {
                    print(lcl$msglist[[i]])
                }
            }
        }
    }
    return(lcl)
}

# localization initialization
setuplocalization = function(domain) {
    # find and bind translation file names
    # domain is the root name of the extension command .R file, e.g., "SPSSINC_BREUSCH_PAGAN"
    # This would be bound to root location/SPSSINC_BREUSCH_PAGAN/lang

    fpath = Find(file.exists, file.path(.libPaths(), paste(domain, ".R", sep="")))
    bindtextdomain(domain, file.path(dirname(fpath), domain, "lang"))
} 
# override for api to account for extra parameter in V19 and beyond
StartProcedure <- function(procname, omsid) {
    if (substr(spsspkg.GetSPSSVersion(),1, 2) >= 19) {
        spsspkg.StartProcedure(procname, omsid)
    }
    else {
        spsspkg.StartProcedure(omsid)
    }
}

gtxt <- function(...) {
    return(gettext(...,domain="STATS_SVM"))
}

gtxtf <- function(...) {
    return(gettextf(...,domain="STATS_SVM"))
}


Run = function(args) {
    #Execute the STATS SVM command

    cmdname = args[[1]]
    args = args[[2]]
    oobj = spsspkg.Syntax(list(
        spsspkg.Template("CMDMODE", subc="", ktype="str", var="cmdmode",
            vallist=list("estimate", "predict")),
        spsspkg.Template("DEPENDENT", subc="",  ktype="existingvarlist", 
            var="dependent"),
        spsspkg.Template("INDEP", subc="", ktype="existingvarlist", var="indep", islist=TRUE),
        spsspkg.Template("ID", subc="", ktype="existingvarlist", var="id"),
        spsspkg.Template("SVMTYPE", subc="", ktype="str", var="svmtype",
            vallist=list("automatic", "cclassification", "nuclassification", "oneclassification",
            "epsregression", "nuregression")),
        spsspkg.Template("KERNEL", subc="", ktype="str", var="kernel",
            vallist=list("linear", "polynomial", "radial", "sigmoid")),
        spsspkg.Template("WORKSPACEFILE", subc="", ktype="literal", var="workspace"),
        
        spsspkg.Template("SCALE", subc="OPTIONS", ktype="bool", var="scale"),
        spsspkg.Template("DEGREE", subc="OPTIONS", ktype="int", var="degree", islist=TRUE,
            vallist=list(1,5)),
        spsspkg.Template("GAMMA", subc="OPTIONS", ktype="float", var="gamma", islist=TRUE),
        spsspkg.Template("COEF0", subc="OPTIONS", ktype="float", var="coef0", islist=TRUE),
        spsspkg.Template("COST", subc="OPTIONS", ktype="float", var="cost", islist=TRUE),
        spsspkg.Template("NU", subc="OPTIONS", ktype="float", var="nu", islist=TRUE),
        spsspkg.Template("EPSILON", subc="OPTIONS", ktype="float", var="epsilon", islist=TRUE),
        spsspkg.Template("CLASSWEIGHTS", subc="OPTIONS", ktype="str", var="classweights", islist=TRUE),
        spsspkg.Template("NUMCROSSFOLDS", subc="OPTIONS", ktype="literal", var="numcrossfolds"),
        spsspkg.Template("PROBPRED", subc="OPTIONS", ktype="bool", var="probpred"),
        spsspkg.Template("MISSING", subc="OPTIONS", ktype="str", var="missingvalues",
            vallist=list("omit", "fail")),
        
        spsspkg.Template("DATASET", subc="SAVE", ktype="varname", var="dataset"),
        spsspkg.Template("WORKSPACE", subc="SAVE", ktype="str", var="workspaceaction",
            vallist=list("retain", "clear")),
        spsspkg.Template("WORKSPACEFILE", subc="SAVE", ktype="literal", 
            var="workspaceoutfile"),
        spsspkg.Template("DECISIONVALUES", subc="SAVE", ktype="bool", var="decisionvalues"),
        
        spsspkg.Template("PLOT", subc="OUTPUT", ktype="bool", var="plotit"),
        spsspkg.Template("FEATUREWEIGHTS", subc="OUTPUT", ktype="bool", var="featureweights")
        
    ))

    # A HELP subcommand overrides all else
    if ("HELP" %in% attr(args,"names")) {
        #writeLines(helptext)
        helper(cmdname)
    }
    else {
        res <- spsspkg.processcmd(oobj, args, "dosvm")
    }
}

helper = function(cmdname) {
    # find the html help file and display in the default browser
    # cmdname may have blanks that need to be converted to _ to match the file
    
    fn = gsub(" ", "_", cmdname, fixed=TRUE)
    thefile = Find(file.exists, file.path(.libPaths(), fn, "markdown.html"))
    if (is.null(thefile)) {
        print("Help file not found")
    } else {
        browseURL(paste("file://", thefile, sep=""))
    }
}
if (exists("spsspkg.helper")) {
assign("helper", spsspkg.helper)
}