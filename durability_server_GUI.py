#!/usr/bin/env python2
######################################################################
# 
# File: durability.py
# 
# Copyright 2018 Backblaze Inc. All Rights Reserved.
# 
######################################################################

import argparse
import math
import sys
import unittest
import tkinter
from tkinter import * 

class Table(object):

    """
    Knows how to display a table of data.

    The data is in the form of a list of dicts:

        [ { 'a' : 4, 'b' : 8 },
          { 'a' : 5, 'b' : 9 } ]

    And is displayed like this:

        |=======|
        | a | b |
        |-------|
        | 4 | 8 |
        | 5 | 9 |
        |=======|
    """

    def __init__(self, data, column_names):
        self.data = data
        self.column_titles = column_names
        self.column_widths = [
            max(len(col), max(len(item[col]) for item in data))
            for col in column_names
        ]

    def __str__(self):
        result = []

        # Title row
        total_width = 1 + sum(3 + w for w in self.column_widths)
        result.append('|')
        result.append('=' * (total_width - 2))
        result.append('|')
        result.append('\n')
        result.append('| ')
        for (col, w) in zip(self.column_titles, self.column_widths):
            result.append(self.pad(col, w))
            result.append(' | ')
        result.append('\n')
        result.append('|')
        result.append('-' * (total_width - 2))
        result.append('|')
        result.append('\n')

        # Data rows
        for item in self.data:
            result.append('| ')
            for (col, w) in zip(self.column_titles, self.column_widths):
                result.append(self.pad(item[col], w))
                result.append(' | ')
            result.append('\n')
        result.append('|')
        result.append('=' * (total_width - 2))
        result.append('|')
        result.append('\n')

        return ''.join(result)

    def pad(self, s, width):
        if len(s) < width:
            return (' ' * (width - len(s))) + s
        else:
            return s[:width]


def print_markdown_table(data, column_names):
    print()
    print(' | '.join(column_names))
    print(' | '.join(['---'] * len(column_names)))
    for item in data:
        print(' | '.join(item[cn] for cn in column_names))
    print()


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def choose(n, r):
    """
    Returns: How many ways there are to choose a subset of n things from a set of r things.

    Computes n! / (r! (n-r)!) exactly. Returns a python long int.

    From: http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    assert n >= 0
    assert 0 <= r <= n

    c = 1
    for num, denom in zip(range(n, n-r, -1), range(1, r+1, 1)):
        c = (c * num) // denom
    return c


def binomial_probability(k, n, p):
    """
    Returns: The probability of exactly k of n things happening, when the
             probability of each one (independently) is p.

    See: https://en.wikipedia.org/wiki/Binomial_distribution#Probability_mass_function
    """
    return choose(n, k) * (p ** k) * ((1 - p) ** (n - k))


class TestBinomialProbability(unittest.TestCase):

    def test_binomial_probability(self):
        # these test cases are from the Wikipedia page
        self.assertAlmostEqual(0.117649, binomial_probability(0, 6, 0.3))
        self.assertAlmostEqual(0.302526, binomial_probability(1, 6, 0.3))
        self.assertAlmostEqual(0.324135, binomial_probability(2, 6, 0.3))

        # Wolfram Alpha: (1 - 1e-6)^800
        self.assertAlmostEqual(0.9992003, binomial_probability(0, 800, 1.0e-6))


def probability_of_failure_for_failure_rate(f):
    """
    Given a failure rate f, what's the probability of at least one failure?
    """
    probability_of_no_failures = math.exp(-f)
    return 1.0 - probability_of_no_failures


def probability_of_failure_in_any_period(p, n):
    """
    Returns the probability that a failure (of probability p in one period)
    happens once or more in n periods.

    The probability of failure in one period is p, so the probability
    of not failing is (1 - p).  So the probability of not
    failing over n periods is (1 - p) ** n, and the probability
    of one or more failures in n periods is:

        1 - (1 - p) ** n

    Doing the math without losing precision is tricky.
    After the binomial expansion, you get (for even n):

        a = 1 - (1 - choose(n, 1) * p + choose(n, 2) p**2 - p**3 + p**4 ... + choose(n, n) p**n)

    For odd n, the last term is negative.

    To avoid precision loss, we don't want to to (1 - p) if p is
    really tiny, so we'll cancel out the 1 and get:
    you get:

        a = choose(n, 1) * p - choose(n, 2) * p**2 ...
    """
    if p < 0.01:
        # For tiny numbers, (1 - p) can lose precision.
        # First, compute the result for the integer part
        n_int = int(n)
        result = 0.0
        sign = 1
        for i in range(1, n_int + 1):
            p_exp_i = p ** i
            if p_exp_i != 0:
                result += sign * choose(n_int, i) * (p ** i)
            sign = -sign
        # Adjust the result to include the fractional part
        # What we want is: 1.0 - (1.0 - result) * ((1.0 - p) ** (n - n_int))
        # Which gives this when refactored:
        result = 1.0 - ((1.0 - p) ** (n - n_int)) + result * ((1.0 - p) ** (n - n_int))
        return result
    else:
        # For high probabilities of loss, the powers of p don't
        # get small faster than the coefficients get big, and weird
        # things happen
        return 1.0 - (1.0 - p) ** n


class TestProbabilityOfFailureAnyPeriod(unittest.TestCase):

    def test_probability_of_failure(self):
        # Easy to check
        self.assertAlmostEqual(0.25, probability_of_failure_in_any_period(0.25, 1))
        self.assertAlmostEqual(0.4375, probability_of_failure_in_any_period(0.25, 2))
        self.assertAlmostEqual(0.0199, probability_of_failure_in_any_period(0.01, 2))

        # From Wolfram Alpha, some tests with tiny probabilities:
        self.assertAlmostEqual(2.0, probability_of_failure_in_any_period(1e-10, 200) * 1e8)
        self.assertAlmostEqual(2.0, probability_of_failure_in_any_period(1e-30, 200) * 1e28)
        self.assertAlmostEqual(7.60690480739, probability_of_failure_in_any_period(3.47347251479e-103, 2190) * 1e100)

        # Check fractional exponents
        self.assertAlmostEqual(0.1339746, probability_of_failure_in_any_period(0.25, 0.5))
        self.assertAlmostEqual(0.0345647, probability_of_failure_in_any_period(0.01, 3.5))


SCALE_TABLE = [
    (1, 'ten'),
    (2, 'a hundred'),
    (3, 'a thousand'),
    (6, 'a million'),
    (9, 'a billion'),
    (12, 'a trillion'),
    (15, 'a quadrillion'),
    (18, 'a quintillion'),
    (21, 'a sextillion'),
    (24, 'a septillion'),
    (27, 'an octillion')
    ]


def pretty_probability(p):
    """
    Takes a number between 0 and 1 and prints it as a probability in
    the form "5 in a million"
    """
    if abs(p - 1.0) < 0.01:
        return 'always'
    for (power, name) in SCALE_TABLE:
        count = p * (10.0 ** power)
        if count >= 0.90:
            return '%d in %s' % (round(count), name)
    return 'NEVER'


def count_nines(loss_rate):
    """
    Returns the number of nines after the decimal point before some other digit happens.
    """
    nines = 0
    power_of_ten = 0.1
    while True:
        if power_of_ten < loss_rate:
            return nines
        power_of_ten /= 10.0
        nines += 1
        if power_of_ten == 0.0:
            return 0

                
def comb_ratio_fails_cause_servers_failures(total_servers, failed_servers, t, s, max_shards_assigned_per_server, min_shards_assigned_per_server, parity_shards):
    """ 
    t = num_of_serveres_with_maxShardsAssignedPerServer
    s = num_of_serveres_with_minShardsAssignedPerServer
    Returns the ratio of pairs that will cause system failure.
    """
    total_comb = choose(total_servers, failed_servers)
    combination_causes_failures = 0
    """
   if 0 server with minShardsAssignedPerServer fail and rest failed are servers with maxShardsAssignedPerServer
   if 1 server with minShardsAssignedPerServer fail and rest failed are servers with maxShardsAssignedPerServer
   Goes until it reaches the num_of_serveres_with_minShardsAssignedPerServer OR servers with minShardsAssignedPerServer = total failed servers
   """
    #print("For ", failed_servers, "Failed Servers =====================================================")
    for f in range(0, s+1, 1): 
        #print(f, "server with minShardsAssignedPerServer and ",failed_servers- f, " with maxShardsAssignedPerServer")
        #proceed only if the rest failures are less than or equal to num_of_serveres_with_maxShardsAssignedPerServer
        if(((failed_servers-f) <= t) & ((failed_servers - f) >= 0)): 
            #print("failed_servers- f <= t OR assigned servers with maxShardsAssignedPerServer < num_of_serveres_with_maxShardsAssignedPerServer")
            # total ssd's in servers with minShardsAssignedPerServer = f * min_shards_assigned_per_server
            # total ssd's in servers with max_shards_assigned_per_server = (failed_servers - f) * max_shards_assigned_per_server
            # if total is greater than parity shards => system fail
            if((f * min_shards_assigned_per_server + (failed_servers - f) * max_shards_assigned_per_server) > parity_shards):
                #print("total-failed_ssd are greater than parity shards: ", f, "*", min_shards_assigned_per_server, "+", (failed_servers - f), "* ", max_shards_assigned_per_server)
                #choose servers with minShardsAssignedPerServer out of num_of_serveres_with_minShardsAssignedPerServer
                #print("combinations cause failures choose(s, f)* choose(t, (failed_servers-f)) = ", choose(s, f) , "*" ,  choose(t, (failed_servers-f) ))
                combination_causes_failures = combination_causes_failures +  choose(s, f) * choose(t, (failed_servers-f))
            """   
            else: 
                print("total-failed_ssd are less than parity shards: ", f, "*", min_shards_assigned_per_server, "+", (failed_servers - f), "* ", max_shards_assigned_per_server)
            
         
        else: 
            print("CAnnot proceed bcz max_shard_assigned_servers are either > num_of_serveres_with_maxShardsAssignedPerServer or < 0")
        """
    print ("For " , failed_servers, " failed_servers, total ", combination_causes_failures, "combinations cause failures out of ", total_comb, " combinations, " , combination_causes_failures, "/", total_comb)
    
    return combination_causes_failures/total_comb

def do_scenario(has_server_input, parity_shards, min_shards, annual_shard_failure_rate, shard_replacement_days, total_servers, annual_server_failure_rate, server_replacement_days):
    """
    Calculates the cumulative failure rates for different numbers of
    failures, starting with the most possible, down to 0.

    The first probability in the table will be extremely improbable,
    because it is the case where ALL of the shards fail.  The next
    line in the table is the case where either all of the shards fail,
    or all but one fail.  The final row in the table is the case where
    somewhere between all fail and none fail, which always happens, so
    the probability is one.
    """
    total_shards = parity_shards + min_shards
    
    print()
    print('#')
    print('# total shards:', total_shards)
    print('# replacement period of a shard(days): %6.4f' % (shard_replacement_days))
    print('# annual shard failure rate: %6.4f' % (annual_shard_failure_rate))
    if(has_server_input):      
        print('# total servers: ', total_servers)      
        print('# replacement period of a server (days): %6.4f' % (server_replacement_days))
        print('# annual server failure rate: %6.4f' % (annual_server_failure_rate))
    print('#')
    print()

    annual_loss_prob = only_shards_cal(parity_shards, min_shards, annual_shard_failure_rate, shard_replacement_days)
    nines_shards = '%d nines' % count_nines(annual_loss_prob)
    #Shard Failures
    print("Annual loss prob (shard failure):" '%10.3e' % annual_loss_prob )
    print("Durability (shard)' :" '%17.15f' % (1.0 - annual_loss_prob))
    
    if(has_server_input):
        annual_loss_prob_server = only_servers_cal(parity_shards, min_shards, total_servers, annual_server_failure_rate, server_replacement_days)
        # Server Failure:
        print("Annual loss prob (server failure):" '%10.3e' % annual_loss_prob_server )
        print("Durability (server)' :" '%17.15f' % (1.0 - annual_loss_prob_server))
        #Probability of server failure P(A or B) = P(A) + P(B) - P(A and B)
        prob_of_shard_AND_server_failure_annual = annual_loss_prob * annual_loss_prob_server
        prob_of_system_failure = annual_loss_prob + annual_loss_prob_server - prob_of_shard_AND_server_failure_annual
        print("Probability of System Failure: ", prob_of_system_failure)
        total_durability = 1.0 - prob_of_system_failure
        nines_total = '%d nines' % count_nines(1- total_durability)
        print("Total Durability: " '%17.15f' %total_durability)
        """
        Total Durability could be find using below formoula
        print("Total Durability (dur1 * dur2) " '%17.15f' % ((1.0 - annual_loss_prob)*(1.0- annual_loss_prob_server)))
        """
    else:
        nines_total = nines_shards
        annual_loss_prob_server = "N/A"
        prob_of_system_failure = "N/A"
        total_durability = "N/A"
    
    print('nines :' , nines_total)
    return annual_loss_prob, annual_loss_prob_server, prob_of_system_failure, total_durability, nines_total

def only_shards_cal(parity_shards, min_shards, annual_shard_failure_rate, shard_replacement_days):
    total_shards = parity_shards + min_shards
    num_periods = 365.0 / shard_replacement_days
    failure_rate_per_period = annual_shard_failure_rate / num_periods
    failure_probability_per_period = 1.0 - math.exp(-failure_rate_per_period)
    period_cumulative_prob = 0.0
    # Cumm Probability for shard failure
    for failed_shards in range(total_shards, total_shards - min_shards, -1):
        period_failure_prob = binomial_probability(failed_shards, total_shards, failure_probability_per_period)
        period_cumulative_prob += period_failure_prob
    annual_loss_prob= probability_of_failure_in_any_period(period_cumulative_prob, num_periods)       
    return annual_loss_prob

    
def only_servers_cal(parity_shards, min_shards, total_servers, annual_server_failure_rate, server_replacement_days):
    total_shards = parity_shards + min_shards
    # server failure in time to repair 
    num_periods_server = 365.0 / server_replacement_days
    failure_rate_per_period_server = annual_server_failure_rate / num_periods_server
    failure_probability_per_period_server = 1.0 - math.exp(-failure_rate_per_period_server)
    period_cummulative_prob_server = 0.0
     # Cumm Prob for server failure
    parity_shards = total_shards - min_shards
    min_shards_assigned_per_server =  total_shards//total_servers
    # number of servers who have more than min_shards_assigned_per_server(num_of_serveres_with_maxShardsAssignedPerServer): total_shards%total_servers
    num_of_serveres_with_maxShardsAssignedPerServer = total_shards%total_servers
    
    # number of servers who have min_shards_assigned_per_server(num_of_serveres_with_minShardsAssignedPerServer) = total_Servers - total_shards%total_servers
    num_of_serveres_with_minShardsAssignedPerServer = total_servers - total_shards%total_servers
 
    if(total_shards%total_servers == 0):
        max_shards_assigned_per_server = min_shards_assigned_per_server
    else:
        max_shards_assigned_per_server = min_shards_assigned_per_server+1
    """    
    print("min_shards_assigned_per_server " , min_shards_assigned_per_server )
    print("max_shards_assigned_per_server ", max_shards_assigned_per_server)
    print("num_of_serveres_with_maxShardsAssignedPerServer: ", num_of_serveres_with_maxShardsAssignedPerServer)
    print("num_of_serveres_with_minShardsAssignedPerServer ", num_of_serveres_with_minShardsAssignedPerServer)
    """
    # l is the number of servers failures that can be survived assumming only servers with min_shards_assigned_per_server are failing 
    # if num_of_serveres_with_minShardsAssignedPerServer = 0, then we can survive all failures of servers who have min_shards_assigned_per_server
    
    if (min_shards_assigned_per_server == 0):
        l = num_of_serveres_with_minShardsAssignedPerServer
    else:
        l = int(parity_shards//min_shards_assigned_per_server)
    #print ("parity_shards//min_shards_assigned_per_server ", parity_shards//min_shards_assigned_per_server)
    # m is the number of servers failures that can be survived assumming only servers with max_shards_assigned_per_server are failing
    m = int(parity_shards//max_shards_assigned_per_server)
    
    #print("l:" ,  l, " m:" , m)
    # Definitely fail if: > l fail
    # No failure fail if <= m fail
    # l > m
    print("SSDs are divided in following way: ")
    for i in  range(1, num_of_serveres_with_maxShardsAssignedPerServer+1, 1):
        print ("Server", i,  "X" * max_shards_assigned_per_server)
    for i in range(num_of_serveres_with_maxShardsAssignedPerServer+1, total_servers+1, 1):
        print("Server", i,  "X" * min_shards_assigned_per_server)
    
    """ 
    for failed_servers in range(total_servers, l, -1):
        period_failure_prob = binomial_probability(failed_servers, total_servers, failure_probability_per_period_server)
        period_cummulative_prob_server += period_failure_prob
    """
    # some combinations of server failure will result in system failures (not necessarily all)
    for failed_servers in range(total_servers, m, -1):
        #If all the servers have equal number of SSDs
        if(min_shards_assigned_per_server == max_shards_assigned_per_server):
            # if failed servers have less than parity shards, then system still runs
            if(min_shards_assigned_per_server * failed_servers < parity_shards):
                failure_ratio = 0
            else:
                failure_ratio = 1
        else: # Some servers have more SSDs than the others
            failure_ratio = comb_ratio_fails_cause_servers_failures(total_servers, failed_servers, num_of_serveres_with_maxShardsAssignedPerServer, num_of_serveres_with_minShardsAssignedPerServer, max_shards_assigned_per_server, min_shards_assigned_per_server, parity_shards)
        period_failure_prob = failure_ratio * binomial_probability(failed_servers, total_servers, failure_probability_per_period_server)
        period_cummulative_prob_server += period_failure_prob
    
    annual_loss_prob_server = probability_of_failure_in_any_period(period_cummulative_prob_server, num_periods_server)
    nines_servers = '%d nines' % count_nines(annual_loss_prob_server)
    
    return annual_loss_prob_server

def gui():
    root = Tk()
    root.title("Durabuility Calculator")
    
    def clear_user_entries():
        dataShard_entry.delete(0, END)
        parityShard_entry.delete(0, END)
        replacementPeriodOfShard_entry.delete(0, END)
        AFRshard_entry.delete(0, END)
        totalServers_entry.delete(0, END)
        replacementPeriodOfServer_entry.delete(0, END)
        AFRserver_entry.delete(0, END)
        
    def clear_output_entries():
        annualLossProbShardFailure_entry.delete(0, END)
        durabilityShard_entry.delete(0, END)
        annualLossProbServerFailure_entry.delete(0, END)
        durability_server_entry.delete(0, END)
        probabilityOfSystemFailure_entry.delete(0, END)
        totalDurability_entry.delete(0, END)
        nines_entry.delete(0, END)
            
    def button_clear():       
        clear_user_entries()
        clear_output_entries()
        
    def pressed_calculate():
        good_entries = True
        global has_server_input
        has_server_input = True
        
        """ Entries to Varibales """
        data_shards = dataShard_entry.get()
        parity_shards = parityShard_entry.get()
        shard_replacement_days = replacementPeriodOfShard_entry.get()
        annual_shard_failure_rate = AFRshard_entry.get()
        total_servers = totalServers_entry.get() 
        server_replacement_days = replacementPeriodOfServer_entry.get()
        annual_server_failure_rate = AFRserver_entry.get()
        
        """ change inputs to int or floats and chck for valid entries"""
        parity_shards = int(parity_shards)
        data_shards = int(data_shards)
        annual_shard_failure_rate = float(annual_shard_failure_rate)
        shard_replacement_days = float(shard_replacement_days)
        
        if(annual_shard_failure_rate <= 0 ):
            AFRshard_entry.insert(0, "Cannot be zero or less ")
            good_entries = False
            
        if(data_shards <= 0):
            dataShard_entry.insert(0, "Cannot be zero or less ")
            good_entries = False
            
        if(parity_shards < 0):
            parityShard_entry.insert(0, "Cannot be less than zero ")
            good_entries = False
       
        if(shard_replacement_days <= 0 ):
            replacementPeriodOfShard_entry.insert(0, "Cannot be zero or less ")
            good_entries = False
        
        #Check if server entries are left empty or have zeros
        if((not total_servers) | (total_servers == "0") | (not server_replacement_days) | (server_replacement_days == "0")| (not annual_server_failure_rate) | (annual_server_failure_rate == "0")):
            has_server_input = False
            
        if(has_server_input):
            total_servers = int(total_servers)
            annual_server_failure_rate = float(annual_server_failure_rate)
            server_replacement_days = float(server_replacement_days)
            if(total_servers <= 0 ):
                totalServers_entry.insert(0, "Cannot be zero or less ")
                good_entries = False
            if(annual_server_failure_rate <= 0):
                AFRserver_entry.insert(0, "Cannot be zero or less ")
                good_entries = False
            if(server_replacement_days <= 0):
                replacementPeriodOfServer_entry.insert(0, "Cannot be zero or less ")
                good_entries = False

        #Call the do_scnerios function to actually do the math
        if(good_entries):
            annual_loss_prob, annual_loss_prob_server, prob_of_system_failure, total_durability, nines_total = do_scenario(has_server_input, parity_shards, data_shards, annual_shard_failure_rate, shard_replacement_days, total_servers, annual_server_failure_rate, server_replacement_days)
        
            """ Add Output to the fields """
            #Clear all output entries before adding somethign else
            clear_output_entries()
            
            #Calculate durabilities & #Print in better format
            durabilityShard = 1.0 - annual_loss_prob
            annual_loss_prob = '%10.3e' % annual_loss_prob
            durabilityShard = '%17.15f' % durabilityShard
            
            # iF server entries are available
            if(has_server_input):
                durability_server = 1.0 - annual_loss_prob_server
                annual_loss_prob_server ='%10.3e' %annual_loss_prob_server
                durability_server = '%17.15f' % durability_server
                prob_of_system_failure = '%17.15f'% prob_of_system_failure
                total_durability = '%17.15f' %total_durability
            else:
                durability_server = "N/A"
            
            annualLossProbShardFailure_entry.insert(0, annual_loss_prob)
            durabilityShard_entry.insert(0, durabilityShard)
            annualLossProbServerFailure_entry.insert(0,  annual_loss_prob_server)
            durability_server_entry.insert(0, durability_server)
            probabilityOfSystemFailure_entry.insert(0, prob_of_system_failure)
            totalDurability_entry.insert(0, total_durability)
            nines_entry.insert(0,nines_total)
    
    """Define Labels """
    dataShard_label = Label(root, text = "Data Shards", anchor="e")
    parityShards_label = Label(root, text = "Parity Shards")
    replacementPeriodOfShard_label = Label(root, text = "Replacement Period of a Shard (Days)")
    AFRshard_label = Label(root, text = "Annual Failure Rate of a Shard")
    totalServers_label = Label(root, text = "Total Servers")
    replacementPeriodOfServer_label = Label(root, text = "Replacment Period of a Server (Days)")
    AFRserver_label = Label(root, text = "Annual Failure rate of a Server")
    
    """ Put Labels on Screen """
    dataShard_label.grid(row = 0, column = 0)
    parityShards_label.grid(row = 1, column = 0)
    replacementPeriodOfShard_label.grid(row = 2, column = 0)
    AFRshard_label.grid(row = 3, column = 0)
    totalServers_label.grid(row = 4, column = 0)
    replacementPeriodOfServer_label.grid(row = 5, column = 0)
    AFRserver_label.grid(row = 6, column = 0)
    
    """ Define User Entries """
    dataShard_entry = Entry (root,  width=40)
    parityShard_entry = Entry(root,  width=40)
    replacementPeriodOfShard_entry = Entry (root,  width=40)
    AFRshard_entry = Entry (root,  width=40)
    totalServers_entry = Entry (root,  width=40)
    replacementPeriodOfServer_entry = Entry (root,  width=40)
    AFRserver_entry = Entry (root,  width=40)
    
    """ Put Entries on Screen"""
    dataShard_entry.grid(row = 0, column = 1)
    parityShard_entry.grid(row = 1, column = 1)
    replacementPeriodOfShard_entry.grid(row = 2, column = 1)
    AFRshard_entry.grid(row = 3, column = 1)
    totalServers_entry.grid(row = 4, column = 1)
    replacementPeriodOfServer_entry.grid(row = 5, column = 1)
    AFRserver_entry.grid(row = 6, column = 1)

    
    """ Define Output Labels """
    annualLossProbShardFailure_label  = Label(root, text = "Annual loss prob (shard failure)")
    durabilityShard_label = Label(root, text = "Durability (shard)")
    annualLossProbServerFailure_label = Label(root, text = "Annual loss prob (server failure)")
    durability_server_label = Label(root, text = "Durability (server)")
    probabilityOfSystemFailure = Label(root, text = "Probability of System Failure")
    totalDurability =  Label(root, text = "Total Durability")
    nines_label  = Label(root, text = "Number of Nines")
        
    """ Display Output Labels """
    annualLossProbShardFailure_label.grid(row = 8, column = 0)
    durabilityShard_label.grid(row = 9, column = 0)
    annualLossProbServerFailure_label.grid(row = 10, column = 0)
    durability_server_label.grid(row = 11, column = 0)
    probabilityOfSystemFailure.grid(row = 12, column = 0)
    totalDurability.grid(row = 13, column = 0)
    nines_label.grid(row = 14, column = 0)
    
    """ Define fields for result """
    annualLossProbShardFailure_entry  = Entry (root, width=40)
    durabilityShard_entry = Entry (root,  width=40)
    annualLossProbServerFailure_entry = Entry (root,  width=40)
    durability_server_entry = Entry (root,  width=40)
    probabilityOfSystemFailure_entry = Entry (root,  width=40)
    totalDurability_entry = Entry (root,  width=40)
    nines_entry = Entry (root,  width=40)
    
    """ Display the fields for results """
    annualLossProbShardFailure_entry.grid(row = 8, column = 1)
    durabilityShard_entry.grid(row = 9, column = 1) 
    annualLossProbServerFailure_entry.grid(row = 10, column = 1)
    durability_server_entry.grid(row = 11, column = 1)
    probabilityOfSystemFailure_entry.grid(row = 12, column = 1)
    totalDurability_entry.grid(row = 13, column = 1)
    nines_entry.grid(row = 14, column = 1) 
    
    """" Default Text to the entries """
    dataShard_entry.insert(0, "Enter Here (only numbers)")
    annualLossProbShardFailure_entry.insert(0, "Results will Display Here")
       
    """ Calculate Button """
    calculate_button = Button(root, text = "Calculate",  width= 30,   command = pressed_calculate)
    calculate_button.grid(row = 7,column = 1)
    
    """ Clear Button """    
    clear_button = Button(root, text = "Clear",  width= 27,  command = button_clear)
    clear_button.grid(row = 7,column = 0)

    root.mainloop()
        
    

if __name__ == '__main__':
    app = gui()

#DO only part of shard if has_server is false inside the function
    

