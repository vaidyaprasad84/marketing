# Heuristics

This module contains class called heuristics. 

The heuristics class contains rule based methods.

# First Touch Attribution

The first touch attribution is model wherein the 100% of the credit is given to the channel that the customer first clicked on before they converted. 

For example, if the customer journey is 

Facebook > Google > Email > Direct > Conversion. The sale credit will to to Facebook. 

# Last Touch Attribution 

The last touch attribution is model wherein 100% of the credit is given to the channel that the customer clicked last before they convered. 

Lets take the same example of customer journey as above.

Facebook > Google > Email > Direct > Conversion. The sale credit in this case will go to Direct Channel.

# Last Touch Attribution w/o channel

This is a special case of last touch attribution. In this if the last clicked channel is something we want to ignore we will ignore that and give credit to previous channel. In most of the cases, the Direct channel is ignored because we want to provide any attribution to only paid channels. 

So if we revisit the journey same as above 

Facebook > Google > Email > Direct > Conversion. The sale credit in this case will go to Email Channel instead of Direct channel. 

However if the journey were: 

Direct > Conversion or Direct > Direct > Converion. Than the sale credit will go to Direct as there is no other channel involved.

# Linear Attribution 
