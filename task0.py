import sys


class HMM_model():
    def __init__(self):
        A,B,pi = self.initialize_parameters()
        state_estimate = self.matrix_multiplication(pi, A)
        observation_probabilities = self.matrix_multiplication(state_estimate, B)
        self.output_vector(observation_probabilities)

    
    def matrix_from_input(self, line):
        line = line.split()
        rows = int(line[0])
        columns = int(line[1])
        matrix_values = list(map(float, line[2:]))
        matrix = []
        
        for row in range(rows):
            first = row * columns
            last = first + columns
            matrix.append(matrix_values[first:last])         
        return matrix
    
    def initialize_parameters(self):
        A = self.matrix_from_input(sys.stdin.readline())
        B = self.matrix_from_input(sys.stdin.readline())
        pi = self.matrix_from_input(sys.stdin.readline())
        pi = pi[0] #Since it only has one object
        return A, B, pi
        
    def matrix_multiplication(self, vector, matrix):
        # 1xn vector
        # And nxm matrix
        n = len(vector)
        m = len(matrix[0])
        output_vector = [] 
        for j in range(m):
            total = 0.0
            for i in range(n):
                total += vector[i] * matrix[i][j]
            output_vector.append(total)
        #Outputs a 1xm vector
        return output_vector
    
    def output_vector(self, vector):
        string = "1 "
        string += str(len(vector))
        string += " "
        for i in range(len(vector)):
            string += str(vector[i])
            string += " "
        print(string)
    
        
        

object = HMM_model()

