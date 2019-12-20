from classification_util import ClassificationUtil

jiho = ClassificationUtil()

jiho.read('typoon.csv')
jiho.show()
jiho.myplot('yearnumber', 'month', 'koreaeffect')
jiho.myviolinplot('month', 'koreaeffect')
jiho.heatmap()
jiho.ignore_warning()
jiho.run_svm(['yearnumber', 'month'], 'koreaeffect')
jiho.run_logistic_regression(['yearnumber', 'month'], 'koreaeffect')
jiho.run_neighbor_classifier (['yearnumber', 'month'], 'koreaeffect', 5)
jiho.run_decision_tree_classifier (['yearnumber', 'month'], 'koreaeffect')