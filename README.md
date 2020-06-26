# invisible data poisonning

These code highlights the possibility to produce an invisible data poisoning: strongly modify the model (resulting from a fair learning) by adding unperceptible perturbation to training samples.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**The code is provided for research only**


## V1 https://www.mdpi.com/2504-4990/1/1/11 - poisonning a frozen deep learning + SVM

**This code use third party**

From theoretic point of view, this code should be fully reproducible.

Yet, it is probably not cross plateform - it has probably hidden dependencies - and it may varies depending on software version, nvidia driver and nvidia hardware.

If you do not manadge to reproduce the experiment, feel free to ask me about more detail.


## V2 - poisonning a deep network (with no data augmentation)

This code has no dependency.

Experiments are not convex anymore, yet, variance is quite low.

If you do not manadge to reproduce the experiment, feel free to ask me about more detail.

## V3 in progress - poisonning a deep network (with data augmentation and torchvision model)

**it seems that the poisoning effect is very sensible to different point like overfitting and data augmentation**

**works in progress**




