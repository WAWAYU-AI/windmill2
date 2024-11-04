#ifndef _KMEANS_2_
#define _KMEANS_2_

class K_Means_2
{
public:
    int max_num;
    int p, q;
    int idx_min, idx_max;
    float value_min, value_max;

    int get_len();
    void get_min_max();
    void update_no_full(float v);
    void update_full(float v);

public:
    float *values;
    float k1, k2;
    int status;

    K_Means_2(int max_num);
    ~K_Means_2();

    void init();
    void update(float v);
};

K_Means_2::K_Means_2(int max_num) : max_num(max_num), p(0), q(0), k1(0), k2(0), status(0)
{
    this->values = new float[this->max_num + 1];
}

K_Means_2::~K_Means_2()
{
    delete this->values;
}

void K_Means_2::init()
{
    this->p = 0;
    this->q = 0;
}

void K_Means_2::update_no_full(float v)
{
    int len = this->get_len();
    this->values[this->q] = v;
    q = (q + 1) % (this->max_num + 1);
    switch (len)
    {
    case 0:
        k1 = v;
        k2 = 0;
        break;

    case 1:

        k1 = this->values[this->p];
        k2 = v;
        break;

    default:
        this->get_min_max();
        int idx;

        k1 = this->value_min;
        k2 = this->value_max;
        while (true)
        {
            float oldk1 = k1;

            int count_k1 = 0, count_k2 = 0;
            float tmp1 = 0, tmp2 = 0;
            for (int i = 0; i <= len; i++)
            {
                idx = (this->p + i) % (this->max_num + 1);
                if (std::abs(values[idx] - k1) < std::abs(values[idx] - k2))
                {
                    count_k1++;
                    tmp1 += values[idx];
                }
                else
                {
                    count_k2++;
                    tmp2 += values[idx];
                }
            }

            count_k1 = count_k1 <= 0 ? 1 : count_k1;
            count_k2 = count_k2 <= 0 ? 1 : count_k2;
            k1 = tmp1 / count_k1;
            k2 = tmp2 / count_k2;

            if (std::abs(oldk1 - k1) < 1e9)
            {
                break;
            }
        }
        break;
    }
    if (this->get_len() >= this->max_num)
    {
        this->status = 1;
    }
}

void K_Means_2::update_full(float v)
{
    p = (p + 1) % (this->max_num + 1);

    this->values[this->q] = v;
    q = (q + 1) % (this->max_num + 1);
    this->get_min_max();

    int idx;
    k1 = this->value_min;
    k2 = this->value_max;
    while (true)
    {
        int count_k1 = 0, count_k2 = 0;
        float tmp1 = 0, tmp2 = 0;
        float oldk1 = k1;
        for (int i = 0; i < this->max_num; i++)
        {
            idx = (this->p + i) % (this->max_num + 1);
            if (std::abs(values[idx] - this->k1) < std::abs(values[idx] - this->k2))
            {
                count_k1++;
                tmp1 += values[idx];
            }
            else
            {
                count_k2++;
                tmp2 += values[idx];
            }
        }
        count_k1 = count_k1 <= 0 ? 1 : count_k1;
        count_k2 = count_k2 <= 0 ? 1 : count_k2;
        k1 = tmp1 / count_k1;
        k2 = tmp2 / count_k2;
        if (std::abs(k1 - oldk1) < 0.01)
        {
            break;
        }
    }
}

void K_Means_2::update(float v)
{
    switch (this->status)
    {
    case 0:
        this->update_no_full(v);
        break;
    case 1:
        this->update_full(v);
        break;
    }
}

void K_Means_2::get_min_max()
{
    int len = this->get_len();
    int idx;
    this->idx_min = this->p;
    this->idx_max = this->p;
    this->value_min = values[this->p];
    this->value_max = values[this->p];
    for (int i = 1; i < len; i++)
    {
        idx = (this->p + i) % (this->max_num + 1);
        if (this->values[idx] > this->values[this->idx_max])
        {
            this->idx_max = idx;
            value_max = values[idx];
        }
        if (this->values[idx] < this->values[this->idx_min])
        {
            this->idx_min = idx;
            value_min = values[idx];
        }
    }
}

int K_Means_2::get_len()
{
    return (this->q - this->p + this->max_num + 1) % (this->max_num + 1);
}

#endif //_KMEANS_2_