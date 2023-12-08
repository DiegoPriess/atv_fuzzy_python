import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
import random


def fcl():
    angulo = ctrl.Antecedent(np.arange(-180, 181, 1), 'angulo')
    angulo_velocidade = ctrl.Antecedent(np.arange(-10, 11, 1), 'angulo_velocidade')
    carro_posicao = ctrl.Antecedent(np.arange(-2, 3, 1), 'carro_posicao')
    carro_velocidade = ctrl.Antecedent(np.arange(-3, 4, 1), 'carro_velocidade')
    carro_forca = ctrl.Consequent(np.arange(-100, 101, 1), 'carro_forca')

    angulo['left'] = fuzz.trapmf(angulo.universe, [-180, -180, -90, -30])
    angulo['zero'] = fuzz.trimf(angulo.universe, [-90, 0, 90])
    angulo['right'] = fuzz.trapmf(angulo.universe, [30, 90, 180, 180])

    angulo_velocidade['left'] = fuzz.trapmf(angulo_velocidade.universe, [-10, -10, -5, 0])
    angulo_velocidade['zero'] = fuzz.trimf(angulo_velocidade.universe, [-5, 0, 5])
    angulo_velocidade['right'] = fuzz.trapmf(angulo_velocidade.universe, [0, 5, 10, 10])

    carro_posicao['left'] = fuzz.trapmf(carro_posicao.universe, [-2, -2, -1, 0])
    carro_posicao['center'] = fuzz.trimf(carro_posicao.universe, [-1, 0, 1])
    carro_posicao['right'] = fuzz.trapmf(carro_posicao.universe, [0, 1, 2, 2])

    carro_velocidade['left'] = fuzz.trapmf(carro_velocidade.universe, [-3, -3, -1.5, 0])
    carro_velocidade['zero'] = fuzz.trimf(carro_velocidade.universe, [-1.5, 0, 1.5])
    carro_velocidade['right'] = fuzz.trapmf(carro_velocidade.universe, [0, 1.5, 3, 3])

    carro_forca['strong_left'] = fuzz.trapmf(carro_forca.universe, [-100, -75, -50, -50])
    carro_forca['left'] = fuzz.trimf(carro_forca.universe, [-75, -50, 0])
    carro_forca['no_push'] = fuzz.trimf(carro_forca.universe, [-10, 0, 10])
    carro_forca['right'] = fuzz.trimf(carro_forca.universe, [0, 50, 75])
    carro_forca['strong_right'] = fuzz.trapmf(carro_forca.universe, [50, 75, 100, 100])

    pendulo_regra_1 = ctrl.Rule(angulo['left'] & angulo_velocidade['left'], carro_forca['right'])
    pendulo_regra_2 = ctrl.Rule(angulo['left'] & angulo_velocidade['zero'], carro_forca['right'])
    carro_regra_1 = ctrl.Rule(carro_posicao['left'] & carro_velocidade['left'], carro_forca['strong_right'])
    carro_regra_2 = ctrl.Rule(carro_posicao['left'] & carro_velocidade['zero'], carro_forca['right'])
    carro_regra_3 = ctrl.Rule(carro_posicao['center'], carro_forca['no_push'])
    carro_regra_4 = ctrl.Rule(carro_posicao['right'] & carro_velocidade['right'], carro_forca['strong_left'])
    carro_regra_5 = ctrl.Rule(carro_posicao['right'] & carro_velocidade['zero'], carro_forca['left'])

    system = ctrl.ControlSystem([pendulo_regra_1, pendulo_regra_2,
                                 carro_regra_1, carro_regra_2, carro_regra_3, carro_regra_4, carro_regra_5])
    controller = ctrl.ControlSystemSimulation(system)

    return controller


def fis():
    controller = fcl()

    tamanho_populacao = 100
    numero_repeticoes = 1000
    probabilidade_de_mutacao = 1
    forca_utilizada = 0

    population = np.random.rand(tamanho_populacao, 4)

    def fitness(individual):
        try:
            controller.input['angulo'] = individual[0] * 360 - 180
            controller.input['angulo_velocidade'] = individual[1] * 20 - 10
            controller.input['carro_posicao'] = individual[2] * 4 - 2
            controller.input['carro_velocidade'] = individual[3] * 6 - 3
            controller.compute()
            force_output = controller.output['carro_forca']
            return abs(force_output - forca_utilizada)
        except ValueError:
            return float('inf')

    for iteration in range(numero_repeticoes):
        selected_parents = []
        for _ in range(len(population)):
            tournament = random.sample(population.tolist(), 2)
            selected_parents.append(min(tournament, key=lambda x: x[3]))
        parents = np.array(selected_parents)
        offsprings = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = random.randint(1, len(parent1) - 2)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offsprings.extend([offspring1, offspring2])
        offsprings = np.array(offsprings)
        mutated_offsprings = offsprings.copy()
        for i in range(len(mutated_offsprings)):
            if random.uniform(0, 100) < probabilidade_de_mutacao:
                mutation_point = random.randint(0, len(mutated_offsprings[i]) - 1)
                mutated_offsprings[i, mutation_point] = np.random.rand()
        population = mutated_offsprings

    resultado_equilibrio_fitness = np.argmin(np.array([fitness(individual) for individual in population]))
    resultado_equilibrio = population[resultado_equilibrio_fitness]
    print("Desenvolvimento do Sistema FIS:", resultado_equilibrio)
    print()
    print()


def generic_fuzzy():
    controller = fcl()

    tamanho_populacao = 100
    numero_repeticoes = 1000
    probabilidade_de_mutacao = 1
    forca_utilizada = 42

    population = np.random.rand(tamanho_populacao, 4)

    def fitness(individual):
        try:
            controller.input['angulo'] = individual[0] * 360 - 180
            controller.input['angulo_velocidade'] = individual[1] * 20 - 10
            controller.input['carro_posicao'] = individual[2] * 4 - 2
            controller.input['carro_velocidade'] = individual[3] * 6 - 3
            controller.compute()
            force_output = controller.output['carro_forca']
            return abs(force_output - forca_utilizada)
        except ValueError:
            return float('inf')

    for iteration in range(numero_repeticoes):
        selected_parents = []
        for _ in range(len(population)):
            tournament = random.sample(population.tolist(), 2)
            selected_parents.append(min(tournament, key=lambda x: x[3]))
        parents = np.array(selected_parents)
        offsprings = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = random.randint(1, len(parent1) - 2)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offsprings.extend([offspring1, offspring2])
        offsprings = np.array(offsprings)
        mutated_offsprings = offsprings.copy()
        for i in range(len(mutated_offsprings)):
            if random.uniform(0, 100) < probabilidade_de_mutacao:
                mutation_point = random.randint(0, len(mutated_offsprings[i]) - 1)
                mutated_offsprings[i, mutation_point] = np.random.rand()
        population = mutated_offsprings

    resultado_equilibrio_fitness = np.argmin(np.array([fitness(individual) for individual in population]))
    resultado_equilibrio = population[resultado_equilibrio_fitness]
    print("Desenvolvimento do Sistema Genético-Fuzzy:", resultado_equilibrio)
    print()
    print()

def neuro_fuzzy():
    controller = fcl()
    mlp = MLPRegressor(hidden_layer_sizes=(5,), max_iter=10000, random_state=42)
    mlp.fit(np.array([[-45, -5, 1, 2]]), np.array([[50]]).ravel())

    angulo = -45
    angulo_velocidade = -5
    carro_posicao = 1
    carro_velocidade = 2

    controller.input['angulo'] = angulo
    controller.input['angulo_velocidade'] = angulo_velocidade
    controller.input['carro_posicao'] = carro_posicao
    controller.input['carro_velocidade'] = carro_velocidade

    controller.compute()

    resultado_equilibrio = mlp.predict(np.array([[angulo, angulo_velocidade, carro_posicao, carro_velocidade]]))[0]

    print("Desenvolvimento do Sistema Neuro-Fuzzy:", resultado_equilibrio)
    print()
    print()


def main():
    fis()
    generic_fuzzy()
    neuro_fuzzy()


if __name__ == "__main__":
    main()
