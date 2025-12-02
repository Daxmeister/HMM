import sys


class HMM_model():
    def __init__(self):
        pass
        
    def run_task_0(self):
        A,B,pi = self.initialize_parameters(0)
        state_estimate = self.multiplication_vector_matrix(pi, A)
        observation_probabilities = self.multiplication_vector_matrix(state_estimate, B)
        self.output_vector(observation_probabilities)
        
    
    def run_task_1(self):
        self.A,self.B,self.pi, self.emissions = self.initialize_parameters(1)

    def initialize_parameters(self, task_number=0):
        A = self.matrix_from_input(sys.stdin.readline())
        B = self.matrix_from_input(sys.stdin.readline())
        pi = self.matrix_from_input(sys.stdin.readline())
        pi = pi[0] #Since it only has one object
        if task_number == 1:
            emissions = self.matrix_from_input(sys.stdin.readline(), True)
            emissions = emissions[0] #Only vector
            return A, B, pi, emissions
        else:
            return A, B, pi
    
    def matrix_from_input(self, line, is_emission=False):
        line = line.split()
        if is_emission: # Since emission vector has it's own input format...........
            rows = 1
            columns = int(line[0])
            matrix_values = list(map(float, line[1:]))
        else:
            rows = int(line[0])
            columns = int(line[1])
            matrix_values = list(map(float, line[2:]))
        matrix = []
        
        for row in range(rows):
            first = row * columns
            last = first + columns
            matrix.append(matrix_values[first:last])         
        return matrix

    

        
    def multiplication_vector_matrix(self, vector, matrix):
        # Mutiplies a 1xn row vector with a nxm matrix
        # Used in task 0
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
object.run_task_0()

