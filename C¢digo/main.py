
from random import random,sample,uniform,randint,gauss
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import FeaturesSelection as FeatureSelection
from Network import Network
from operator import itemgetter
import time



def func(x):
    if x ==0:
        return 0
    return 1

def read_data():
     #Features Descriptions
    
    dt = pd.read_csv('processed.cleveland.data', sep=',', encoding='ISO-8859-1', engine='python',header=None)
    dt.columns=['Age','Sex','CP','Trestbps','Chol','Fbs','restecg','Thalach','exang','oldpeack','slope','ca','thal','num']
    
    dt = dt.replace('?', np.nan)
    dataset=dt.dropna(how='any')
    dataset['ca'] = dataset['ca'].astype(float)
    dataset['thal'] = dataset['thal'].astype(float)
    dataset['num'] = dataset['num'].apply(func)

    target=dataset['num']
    del dataset['num']
    
    
    '''-------------------------Feature Selection-----------------------------'''
    # Correlation Matriz
    #FeatureSelection.CorrelationMatrizWithHeatMap(dataset, target)
    n_features = 9
    dataset_select_Uni = FeatureSelection.Univariate_Selection(dataset, target, n_features)
    dataset_select = FeatureSelection.Feature_Importance(dataset, target, n_features)
    
    '''-------------------------Divide Dataset into Train and Test-----------------------------'''
    data_train, data_test, target_train, y_test = train_test_split(dataset_select, target, test_size=0.2,
                                                                        random_state=0)
    data_train = data_train.reset_index(drop=True)
    y_train = target_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    '''-------------------------Scaling-----------------------------'''
    colunms = list(data_train)
    # data_train_scaled, data_test_scaled = Scaling.Scaling_StandardScaler(data_train, data_test, colunms)
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    X_train = scaler.transform(data_train)
    X_test = scaler.transform(data_test)
    #data_train_scaled, data_test_scaled = Scaling.Scaling_RobustScaler(data_train, data_test, colunms)

    return X_train,y_train,X_test,y_test

def get_all_scores(X,y,nets):
    return [net.mean_error(X, y) for net in nets]

def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population,size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]

def get_random_point(type,nets):

        '''
        @type = either 'weight' or 'bias'
        @returns tuple (layer_index, point_index)
            note: if type is set to 'weight', point_index will return (row_index, col_index)
        '''

        nn = nets[0]
        layer_index, point_index = randint(0, nn.num_layers-2), 0
        if type == 'weight':
            row = randint(0,nn.weights[layer_index].shape[0]-1)
            col = randint(0,nn.weights[layer_index].shape[1]-1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = randint(0,nn.biases[layer_index].size-1)
        return (layer_index, point_index)


def mutation(child,mut_rate,nets):


        nn = copy.deepcopy(child)


        """
        # mutate bias
        for _ in range(nets[0].bias_nitem):
            # get some random points
            layer, point = get_random_point('bias',nets)
            # add some random value between -0.5 and 0.5
            if uniform(0,1) < mut_rate:
                nn.biases[layer][point] += uniform(-0.5, 0.5)

        # mutate weight
        for _ in range(nets[0].weight_nitem):
            # get some random points
            layer, point = get_random_point('weight',nets)
            # add some random value between -0.5 and 0.5
            #print(layer,point[0],point[1])
            #print(nn.weights[layer])
            #print(nn.weights[layer][point[0]][point[1]])
            if uniform(0,1) < mut_rate:
                nn.weights[layer][point[0]][point[1]] += uniform(-0.5, 0.5)"""

        for i in range(len(nets[0].weights)):
            for j in range(len(nets[0].weights[i])):
                for k in range(len(nets[0].weights[i][j])):
                    value = random()
                    if(value < mut_rate):
                        muta_value = gauss(0,1)
                        new_gene = nn.weights[i][j][k] + muta_value
                        """if new_gene < 0: # Estes "2" sao random podem tentar outro numero
	                        new_gene = 0
                        elif new_gene > 1:
                            new_gene = 1"""
                        nn.weights[i][j][k] = new_gene

        for i in range(len(nets[0].biases)):
            for j in range(len(nets[0].biases[i])):
                value = random()
                if(value < mut_rate):
                    muta_value = gauss(0,1)
                    new_gene = nn.biases[i][j] + muta_value
                    """if new_gene < 0: # Estes "2" sao random podem tentar outro numero
	                    new_gene = 1
                    elif new_gene > 1:
                        new_gene = 1"""
                    nn.biases[i][j] = new_gene




        return nn

def crossover(father, mother,cross_rate,nets):

        """
        @father = neural-net object representing father
        @mother = neural-net object representing mother
        @returns = new child based on father/mother genetic information
        """


        value = random()
        if(value < cross_rate):
            child1 = copy.deepcopy(father[0])
            child2 = copy.deepcopy(mother[0])
            cromo1 = father[0]
            cromo2 = mother[0]
            for i in range(len(nets[0].weights)):
                for j in range(len(nets[0].weights[i])):
                    for k in range(len(nets[0].weights[i][j])):
                        value = random()
                        if(value < 0.5):
                            child1.weights[i][j][k] = cromo2.weights[i][j][k]
                            child2.weights[i][j][k] = cromo1.weights[i][j][k]

            for i in range(len(nets[0].biases)):
                for j in range(len(nets[0].biases[i])):
                    value = random()
                    if(value < 0.5):
                        child1.biases[i][j] = cromo2.biases[i][j]
                        child2.biases[i][j] = cromo1.biases[i][j] 
            return ((child1,0),(child2,0))
        else:
            return (father,mother)

            
def one_point_cross(indiv_1, indiv_2,prob_cross):
	value = random()
	if value < prob_cross:
	    cromo_1 = indiv_1[0]
	    cromo_2 = indiv_2[0]
	    pos = randint(0,len(cromo_1))
	    f1 = cromo_1[0:pos] + cromo_2[pos:]
	    f2 = cromo_2[0:pos] + cromo_1[pos:]
	    return ((f1,0),(f2,0))
	else:
	    return (indiv_1,indiv_2)

def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        new_population = [pop[0] for pop in new_population]
        return new_population
    return elitism

def GeneticAlgo(best_ele,fitness,run,X_train,y_train,X_test,y_test,gen_size,pop_size,net_sizes,mut_rate,cross_rate,sel_parents,recombination,sel_survivors):

    file_name = str(run) + ".txt"

    newfile = open(file_name,"w")

    start_time = time.time()

    # Init pop_size diferent Networks
    pop = [Network(net_size) for i in range(pop_size)]

    #evaluate the population
    score_list = list(zip(pop, get_all_scores(X_train,y_train,pop)))

    for j in range(gen_size):

        mate_pool = sel_parents(score_list)

        progenitores = []
        for i in  range(0,pop_size-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, cross_rate,pop)
            progenitores.extend(filhos) 
        
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,mut_rate,pop)
            descendentes.append(novo_indiv)

        score_descendentes = list(zip(descendentes, get_all_scores(X_train,y_train,descendentes)))

        pop = sel_survivors(score_list,score_descendentes)

        score_list = list(zip(pop, get_all_scores(X_train,y_train,pop)))
        score_list.sort(key=itemgetter(1))

        """print("Current iteration : {}".format(j+1))
        print("Time taken by far : %.1f seconds" % (time.time() - start_time))
        print("Current top member's network accuracy: %.2f%%\n" % score_list[0][0].accuracy(X_train,y_train))
        print("Mean Error:", score_list[0][0].mean_error(X_train,y_train))"""

        if(fitness > score_list[0][1]):
            fitness = score_list[0][1]
            best_ele = score_list[0][0]
            print(fitness)

        newfile.write(str(score_list[0][0].accuracy(X_train,y_train)) + "\n")


        """print("\n")
        print(score_list[0][0])
        print("\n")"""
    newfile.close()

    return fitness,best_ele


if __name__ == "__main__":
    
    X_train, y_train, X_test, y_test = read_data()

    pop_size = 50
    gen_size = 100
    mut_rate = 0.05
    crossover_rate = 0.9
    elit_rate = 0.3
    net_size = [9,100,50,20,2]
    fitness = [99999999999,None]

    #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    for i in range(30):
        fitness = GeneticAlgo(fitness[1],fitness[0],i,X_train,y_train,X_test,y_test,gen_size,pop_size,net_size,mut_rate,crossover_rate,tour_sel(3),crossover,sel_survivors_elite(elit_rate))

    print("Best Network form all the runs: ", fitness[1].accuracy(X_test,y_test))