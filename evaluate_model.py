import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# تحميل البيانات والموديل
df = pd.read_csv('/home/ubuntu/featured_data.csv')
model = joblib.load('/home/ubuntu/car_price_model.pkl')
features = joblib.load('/home/ubuntu/model_features.pkl')

X = df[features]
y = df['Price']

# التنبؤ على كامل البيانات (لأغراض العرض فقط)
y_pred = model.predict(X)

# المقاييس
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# رسم المقارنة بين القيم الحقيقية والمتوقعة
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.savefig('/home/ubuntu/evaluation_plot.png')

# أهمية المميزات (Feature Importance)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('/home/ubuntu/feature_importance.png')
