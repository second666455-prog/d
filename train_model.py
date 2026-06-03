import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# تحميل البيانات
df = pd.read_csv('/home/ubuntu/featured_data.csv')

# تحديد الميزات (X) والهدف (y)
X = df.drop('Price', axis=1)
y = df['Price']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. تدريب Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# 2. تدريب Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# تقييم سريع
print("Linear Regression R2:", r2_score(y_test, lr_preds))
print("Random Forest R2:", r2_score(y_test, rf_preds))

# حفظ الموديل الأفضل (سنختار Random Forest عادة لمرونته، أو حسب النتائج)
# بما أن البيانات صغيرة جداً، قد لا تكون النتائج دقيقة تماماً، لكننا سنكمل العملية
joblib.dump(rf_model, '/home/ubuntu/car_price_model.pkl')
joblib.dump(X.columns.tolist(), '/home/ubuntu/model_features.pkl')

print("Model and features saved successfully.")
