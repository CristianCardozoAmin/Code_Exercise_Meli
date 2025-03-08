
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
    log_loss, roc_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def preprocess_dataframe_model(df):
    """
    """
    num_features = df.select_dtypes(include=['number']).columns
    cat_features = df.select_dtypes(include=['object', 'category']).columns
    bool_features = df.select_dtypes(include=['bool']).columns

    # Num ericos 
    if len(num_features) > 0:
        scaler = StandardScaler()
        scaled_numeric_df = pd.DataFrame(
            scaler.fit_transform(df[num_features]),
            columns=num_features,
            index=df.index
        )
    else:
        scaled_numeric_df = pd.DataFrame(index=df.index)

    #Categoriass 
    if len(cat_features) > 0:
        dummy_cat_df = pd.get_dummies(df[cat_features], drop_first=True)
    else:
        dummy_cat_df = pd.DataFrame(index=df.index)

    #Boleanas
    if len(bool_features) > 0:
        bool_df = df[bool_features].astype(int)
    else:
        bool_df = pd.DataFrame(index=df.index)

    #Retornamos la informacion unida
    return pd.concat([scaled_numeric_df, dummy_cat_df, bool_df], axis=1)

def create_model_and_save(X_train, y_train, ruta_modelo='../data/models/xgb_model_gen.pkl'):
    """
    """
    param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [43, 87, 150]
        }

    xgb_model = xgb.XGBClassifier(
        tree_method="hist",
        random_state=20250307
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10
    )

    grid_search.fit(X_train, y_train)
    modelo_optimizado = grid_search.best_estimator_
    joblib.dump(modelo_optimizado, ruta_modelo)

    return modelo_optimizado

def create_classification_metrics(y_true, y_pred, y_prob=None):
    """
    """
    metrics = {}

    print("Metricas basicas...")
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    print("matrix...")
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            metrics['roc_auc'] = f'Error calculando ROC-AUC: {e}'
        
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except Exception as e:
            metrics['log_loss'] = f'Error calculando log_loss: {e}'
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        metrics['roc_auc_curve'] = roc_auc
    else:
        metrics['roc_auc'] = 'y_prob no proporcionado'
        metrics['log_loss'] = 'y_prob no proporcionado'
        metrics['roc_curve'] = None
        metrics['roc_auc_curve'] = None

    for key, value in metrics.items():
        print(f"{key}:\n{value}\n")

    return 0

def show_importance_variables(X_train, model):
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp = imp[imp > 0].sort_values(ascending=False).head(15)
    imp.plot.bar()
    plt.ylabel('Importancia')
    plt.title('Importancia de Variables')
    plt.show()     

def evaluate_model(X_test,y_test,model):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  
    create_classification_metrics(y_test, y_pred, y_prob=y_pred_proba)




