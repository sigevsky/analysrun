# with db.session() as session:
#     with session.begin_transaction() as get_data:
#         result = get_data.run()
#         training_data = pd.DataFrame([{k: v for k, v in r.items()} for r in result])
#
#         features_columns = [x for x in training_data.columns if x not in ['use_id', 'book_id']]
#         responses_columns = ['interested_in']
#         X_data = training_data[features_columns]
#         Y_data = training_data[responses_columns]
#         X_data.head()
#         y = Y_data['interested_in'].values
#
#         new_data = pd.DataFrame(data={'x1': [10], 'x2': [8]})
#
#         model_1 = LogisticRegression()
#         model_1.fit(X_data, y)
#         model_1.predict(new_data)