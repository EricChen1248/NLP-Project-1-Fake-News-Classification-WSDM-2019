import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

new_train = pd.read_csv("./combined.csv", index_col=0)
new_test = pd.read_csv("./combined_test.csv", index_col=0)
train = pd.read_csv("./feature_train.csv")
test = pd.read_csv("./feature_test.csv")

FEATURE_COLUMNS = [""]

train['EnglishNounScore'] = new_train['EnglishNounScore']
train['EmbeddingScore'] = new_train['EmbeddingScore']


print(train.shape[0])

test['EnglishNounScore'] = new_test['EnglishNounScore']
test['EmbeddingScore'] = new_test['EmbeddingScore']

print(test.shape[0])
train.to_csv("./concat_train.csv", index=False)
test.to_csv("./concat_test.csv", index=False)


# new_trian = new_train.as_matrix()
# new_test = new_test.as_matrix()
# train = train.as_matrix()
# test = test.as_matrix()

# concat_train = np.concatenate((train, new_train), axis=1)
# concat_test = np.concatenate((test, new_test), axis=1)

# concat_train = pd.DataFrame(concat_train, columns=)



# print(concat_train)
# print(concat_test)

