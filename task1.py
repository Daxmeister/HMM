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
        A,B,pi, emissions = self.initialize_parameters(1)
        value = self.forward_algorithm(A, B, pi, emissions)
        print(value)
        
    def run_task_2(self):
        A,B,pi, emissions = self.initialize_parameters(2)
        probable_emissions = self.viterbi_algorithm(A, B, pi, emissions)
        self.output_nonvector(probable_emissions)
        

        
    def initialize_parameters(self, task_number=0):
        A = self.matrix_from_input(sys.stdin.readline())
        B = self.matrix_from_input(sys.stdin.readline())
        pi = self.matrix_from_input(sys.stdin.readline())
        pi = pi[0] #Since it only has one object
        if task_number == 1 or task_number == 2:
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
            matrix_values = list(map(int, line[1:])) # emission has integers, not floats
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
    
    def element_wise_product(self, a, b):
        # Element wise product of two 1xn vectors (lists)
        c = []
        for index in range(len(a)):
            c.append(a[index]*b[index])
        return c
    
    def column_from_B(self, B, observation):
        # Outputs a the column of state probabilities given observation
        # Note that observation must be the "index"
        output_vector = []
        for row_number in range(len(B)):
            output_vector.append(B[row_number][observation])
        return output_vector
    

    
    def forward_algorithm(self, A,B,pi,emissions):
        alpha = []
        
        #Step 1 - Initialize
        B_column = self.column_from_B(B, emissions[0])
        alpha = self.element_wise_product(pi, B_column)
        
        # Step 2 - Middle part
        for emission_number in range(1, len(emissions)):
            B_column = self.column_from_B(B, emissions[emission_number])
            state_probabilities = self.multiplication_vector_matrix(alpha, A)
            alpha = self.element_wise_product(state_probabilities, B_column)
        
        # Step 3 - Final step
        total = 0
        for value in alpha:
            total += value
        return total
    
    def viterbi_algorithm(self, A, B, pi, emissions):
        delta = []
        previous_states = []
        n_states = len(A)
        
        # Step 1 - initialize
        
        emission_number = emissions[0]
        B_column = self.column_from_B(B, emission_number)
        delta.append(self.element_wise_product(pi, B_column))
        previous_states.append([None]*n_states)
        
        # Step 2 - Iterate
        
        # For every emission after the first, calculate delta
        for n in range(1, len(emissions)): 
            delta_n, previous_state_n = self.viterbi_step_2(A, B, emissions, n, n_states, delta)
            delta.append(delta_n)
            previous_states.append(previous_state_n)

        
        # Step 3 - Terminate
        
        probable_states = []
        
        # Calculate most likely state at end
        i = len(emissions)-1 #
        most_probable_state_at_i_minus_1 = delta[i].index(max(delta[i])) 
        probable_states.append(most_probable_state_at_i_minus_1)
        
        # Calculate all preceding states
        for i in range(len(emissions)-1, 0, -1): 
            most_probable_state = most_probable_state_at_i_minus_1 # This line exists for pedagogical purposes
            most_probable_state_at_i_minus_1 = previous_states[i][most_probable_state] # Identifies most likely state at time i-1
            probable_states.append(most_probable_state_at_i_minus_1) 
        
        # Reverse order 
        probable_states.reverse()
        return probable_states
                    
        
    
    def viterbi_step_2(self, A, B, emissions, n, n_states, delta):
        emission = emissions[n]
        delta_n=[]
        most_likely_previous_states_n=[]
        for state_i in range(n_states):
            max_probability = 0 
            most_likely_previous_state = None
            for state_j in range(n_states):
                value = delta[n-1][state_j] * A[state_j][state_i] * B[state_i][emission]
                if value > max_probability:
                    max_probability = value
                    most_likely_previous_state = state_j
            delta_n.append(max_probability)
            most_likely_previous_states_n.append(most_likely_previous_state)
        return delta_n, most_likely_previous_states_n
    
            
    
    def output_vector(self, vector):
        string = "1 "
        string += str(len(vector))
        string += " "
        for i in range(len(vector)):
            string += str(vector[i])
            string += " "
        print(string)
    
    def output_nonvector(self, vector):
        out_string = ""
        for element in vector:
            out_string = out_string + " " + str(element)
        print(out_string)
    
    
        
        

object = HMM_model()
object.run_task_2()
#object.run_task_0()

